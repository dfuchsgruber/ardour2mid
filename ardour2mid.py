#!/usr/bin/env python3

# Mostly based on https://github.com/dbolton/ArdourMIDIExport/blob/master/main.py

import os
import argparse
import typing
import warnings
from collections import defaultdict

import xml.dom.minidom
import mido
import numpy as np
import re
from typing import Tuple, Dict, List

PITCH_BENDER = 'pitch-bender'

def verbose_print(*args, verbose : bool = False, **kwargs):
    if verbose:
        print(*args, **kwargs)

def get_sources(ardour_dom : xml.dom.minidom.Document, midi_dir : str, verbose: bool=False) -> Tuple[Dict[str, str], Dict[str, Dict[int, str]]]:
    """ Gets all midi sources references by a ardour project
    @param ardour_dom: The ardour xml project tree
    @param midi_dir: Where to find the source midi files.
    @param verbose: if verbose output will be printed
    @return: mapping from source ids to source names
    @return: mapping from source ids to the automation names
    """
    sources = {}
    automation_interpolation_styles = defaultdict(dict)

    for source in ardour_dom.getElementsByTagName('Source'):
        if source.getAttribute('type') == 'midi':
            path = os.path.join(midi_dir, source.getAttribute('name'))
            if not os.path.exists(path):
                # warnings.warn(f'Could not find midi source {path}', RuntimeWarning)
                pass
            else:
                source_id = source.getAttribute('id')
                sources[source_id] = path
                for interpolation_style in source.getElementsByTagName('InterpolationStyle'):
                    name = interpolation_style.getAttribute('parameter')
                    if m := re.match(r"midicc-([0-9]+)-([0-9]+)", name):
                        channel, control = int(m.group(1)), int(m.group(2))
                        if channel == 0:
                            automation_interpolation_styles[source_id][control] = interpolation_style.getAttribute('style').lower()
                    elif m := re.match(r"midi-pitch-bender-([0-9]+)", name):
                        channel = int(m.group(1))
                        if channel == 0:
                            automation_interpolation_styles[source_id][PITCH_BENDER] = interpolation_style.getAttribute('style').lower()

    for source_id in automation_interpolation_styles:
        verbose_print(f'Found midi control change automation interpolation styles for {source_id}: {automation_interpolation_styles[source_id]}', verbose=verbose)

    return sources, automation_interpolation_styles

def midi_create_meta_track(ardour_dom : xml.dom.minidom.Document, output_midi : mido.MidiFile, ticks_per_beat: int, sample_rate: int, verbose : bool = False) -> mido.MidiTrack:
    """ Creates a meta track containing tempo, position markers etc. """
    meta_track = output_midi.add_track(name='meta track')

    # Tempo
    tempo = ardour_dom.getElementsByTagName('Tempo')[0]
    bpm = int(tempo.getAttribute('beats-per-minute'))
    tempo = mido.bpm2tempo(bpm)
    meta_track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    marker_times, markers = [], []
    
    for location in ardour_dom.getElementsByTagName('Location'):
        if location.getAttribute('flags') == 'IsMark':
            # Add this position marker
            name = location.getAttribute('name')
            start = int(location.getAttribute('start'))
            end = int(location.getAttribute('end'))
            if start != end:
                warnings.warn(f'Position marker {name} has different start and end positions. Setting it at start position.')
            start_ticks = int(np.round(mido.second2tick(start / sample_rate, ticks_per_beat, tempo), decimals=0))
            markers.append(name)
            marker_times.append(start_ticks)

    idx_sorted = np.argsort(marker_times)
    marker_times, markers = np.array(marker_times)[idx_sorted], np.array(markers)[idx_sorted]
    marker_times[1:] -= marker_times[:-1] # Calculate differences between marker positions

    for time, marker in zip(marker_times, markers):
        msg = mido.MetaMessage('marker', time=time, text=marker)
        meta_track.append(msg)

    verbose_print(f'Added meta track to output midi.', verbose=verbose)

    return meta_track

def linearize_control(ts: np.ndarray, vs: np.ndarray, increment: int=1) -> Tuple[np.ndarray, np.ndarray]:
    """ Linearlizes a sequence of controller events. 
    
    @param ts: time of the control events
    @param vs: value of the control events
    @param increment: at which increments to add additional control events
    @return ts: time of additional control events
    @return vs: values of additional control events
    """
    dvs, dts = vs[1:] - vs[:-1], ts[1:] - ts[:-1]
    signs, steps = np.sign(dvs), np.abs(dts / dvs)

    additional_ts, additional_vs = [], []
    for idx in range(dvs.shape[0] - 1):
        if dvs[idx] != 0 and dts[idx] != 0:
            for j in range(1, np.abs(dvs[idx]), increment): # Don't add the endpoints ts[i] and ts[i + 1]
                additional_ts.append(ts[idx] + j * steps[idx])
                additional_vs.append(vs[idx] + j * signs[idx])

    return np.array(additional_ts).astype(int), np.array(additional_vs)

def track_add_messages(track: mido.MidiTrack, messages: List[mido.Message], automation_interpolation_styles: dict, verbose: bool = False,
                        control_change_resolution: int = 1, pitch_bend_resolution:int = 100):
    """ Adds messages to a track. If an midi control change automation uses linear interpolation, additional control change events
        will be inserted that mimic a (discretized) version of the linear curve.
        
    @param track: The track to add messages to
    @param messages: all messages to add
    @param automation_inerpolation_styles: interpolation styles for all midi control change automations. Defaults to "linear",
                                           if a control number is not present.
    @param verbose: If verbose output will be printed.
    @param control_change_resolution: at which resolution control change events will be linearized
    @param pitch_bend_resolution: at which resolution pitch bend events will be linearized
    """

    # Set absolute times of all events, easier to work with
    abs_times = np.cumsum([m.time for m in messages])
    for m, abs_time in zip(messages, abs_times):
        m.time = abs_time
    
    # Find all control change messages in the track with linear interpolation style
    # Imitate the linear control change by adding control change messages whenever the
    # value would change
    linear_ccs = set(m.control for m in messages if m.type == 'control_change' and automation_interpolation_styles.get(m.control, 'linear') == 'linear')
    additional_cc_msgs = []
    for cc in linear_ccs:
        cc_msgs = [m for m in messages if m.type == 'control_change' and m.control == cc]
        assert all(m.channel == 0 for m in messages if m.type == 'control_change' and m.control == cc)
        
        ts, vs = linearize_control(np.array([m.time for m in cc_msgs]), np.array([m.value for m in cc_msgs]), increment=control_change_resolution)
        for t, v in zip(ts, vs):
            additional_cc_msgs.append(mido.Message(
                'control_change', channel=0, control=cc, value=v, time=t,
            ))

    # Find pitch-bending messages
    pitch_bend_msgs = [m for m in messages if m.type == 'pitchwheel' and m.channel == 0]
    if len(pitch_bend_msgs) > 1:
        ts, vs = linearize_control(np.array([m.time for m in pitch_bend_msgs]), np.array([m.pitch for m in pitch_bend_msgs]), increment=pitch_bend_resolution)

        for t, v in zip(ts, vs):
            additional_cc_msgs.append(mido.Message(
                'pitchwheel', channel=0, pitch=v, time=t,
            ))

    all_messages = sorted(messages + additional_cc_msgs, key=lambda msg: msg.time)
    rel_times = np.array([m.time for m in all_messages])
    rel_times[1:] -= rel_times[:-1]
    assert (rel_times >= 0).all()

    # Re-apply relative times to the events
    for rel_time, msg in zip(rel_times, all_messages):
        msg.time = rel_time

    for m in all_messages:
        track.append(m)
        
    verbose_print(f'Added {len(all_messages)} messages to track.', verbose=verbose)

def sanitize_messages(messages: Dict[str, List[mido.Message]]) -> Dict[str, List[mido.Message]]:
    """ Sanitizes the messages in a playlist. """
    sanitized = {}
    for route_id, msgs in messages.items():
        sanitized[route_id] = []
        note_is_on = [False for _ in range(128)]
        for msg in msgs:
            if msg.type == 'note_on':
                if note_is_on[msg.note]:
                    warnings.warn(f'Overlapping notes in route {route_id} at tick {msg.time}. Turning off previous note.')
                    sanitized[route_id].append(mido.Message('note_off', note=msg.note, time=msg.time))
                    msg = msg.copy()
                    msg.time = 0
                note_is_on[msg.note] = True
            elif msg.type == 'note_off':
                if not note_is_on[msg.note]:
                    warnings.warn(f'Found note off event without note on event in route {route_id} at {msg.time}. Skipping...')
                    continue
                note_is_on[msg.note] = False
            sanitized[route_id].append(msg)
        if any(note_is_on):
            warnings.warn(f'Notes are still on at the end of route {route_id}. Is this wanted?')
    return sanitized

def assemble(ardour_dom : xml.dom.minidom.Document, output_midi : mido.MidiFile, midi_sources : Dict[str, str],
                messages : Dict[str, List[mido.Message]], ticks_per_beat=19200, verbose : bool = False):
    """ Assembles all regions into a single midi file with multiple tracks.
    
    @param ardour_dom: The ardour document tree
    @param output_midi: The output midi file to assemble into
    @param midi_sources: Mapping from midi source name to the path of the source
    @param ardour_route_id_to_midi: Mapping from ardour route id to midi track
    @param ticks_per_beat: How many ticks there are per beat (used for tempo)
    @param verbose: If verbose output will be logged.
    """

    for playlist_idx, playlist in enumerate(ardour_dom.getElementsByTagName('Playlist')): # Each playlist will be one track
        verbose_print(f'Iterating playlist {playlist_idx}...', verbose=verbose)
        if playlist.getAttribute('type') == 'midi':
            route_id = playlist.getAttribute('orig-track-id')
            if not route_id in messages:
                verbose_print(f'\tCould not match playlist {playlist_idx} to a route (key {route_id} missing).\n\tSkipping.', verbose=verbose)
                continue
            # track = ardour_route_id_to_midi[route_id]
            verbose_print(f'\tMatched to route {route_id}', verbose=verbose)
            
            # Iterate through all the regions
            total_time = 0
            previous_region_end_time = 0
            for region_idx, region in enumerate(playlist.getElementsByTagName('Region')):
                verbose_print(f'\tRegion {region_idx}:', verbose=verbose)
                source_midi_path = midi_sources.get(region.getAttribute('source-0'), None)
                # print(source_midi_path)
                if source_midi_path:
                    source_midi = mido.MidiFile(source_midi_path)
                    start_beats, length_beats, beat = (float(region.getAttribute(attr)) for attr in ('start-beats', 'length-beats', 'beat'))

                    # Keep track of which note is on and off. We want to set notes to "off" if they are playing after a region has ended
                    note_is_on = [False for _ in range(128)]

                    total_time += beat * ticks_per_beat - previous_region_end_time
                    source_midi_total_time = 0
                    num_events_inserted = 0

                    for msg in source_midi.tracks[0]:
                        if not msg.is_meta:
                            source_midi_total_time += msg.time
                            time_first_msg = int((total_time - previous_region_end_time) + (source_midi_total_time - start_beats * ticks_per_beat))
                            if source_midi_total_time >= start_beats * ticks_per_beat and source_midi_total_time <= (length_beats + start_beats) * ticks_per_beat:
                                if msg.type == 'note_on':
                                    note_is_on[msg.note] = True
                                elif msg.type == 'note_off':
                                    note_is_on[msg.note] = False
                                if time_first_msg < 0:
                                    raise RuntimeWarning(f'Region {region_idx} in playlist {playlist_idx} overlapps with previous region.')
                                elif num_events_inserted == 0:
                                    msg_to_insert = msg.copy(time = time_first_msg)
                                    total_time = time_first_msg + previous_region_end_time
                                else:
                                    msg_to_insert = msg.copy(time = msg.time)
                                    total_time += msg.time
                                if source_midi_total_time < (length_beats + start_beats) * ticks_per_beat or msg.type != 'note_on':
                                    messages[route_id].append(msg_to_insert)
                                    num_events_inserted += 1
                                else:
                                    verbose_print(f'\t\tIgnored midi event at {msg.time}', verbose=verbose)
                            else:
                                # Turn off all nodes that are still playing after the region ended
                                for note in range(128):
                                    if note_is_on[note]:
                                        messages[route_id].append(mido.Message('note_off', note=note))
                                        note_is_on[note] = False
                                        verbose_print(f'\t\tNote {note} was playing at end of region. Insert note_off event.', verbose=verbose)
                        
                    verbose_print(f'\t\tInserted {num_events_inserted} midi events.', verbose=verbose)
                    previous_region_end_time = total_time
                else:
                    raise RuntimeError(f'\tRegion midi not found. Could not parse as it might get wrong timings.')

def export(input_file : str, output_file : str, create_meta_track : bool = True, ticks_per_beat : int = 19200, verbose : bool = False,
            control_change_resolution: int = 4, pitch_bend_resolution: int = 256):
    """ Merges and exports the midi files of a project. 
    @param input_file: Input adrour project file
    @param output_file: Output midi file
    @param create_meta_track: If to create a meta track for loops, tempo etc.
    @param ticks_per_beat: How many ticks there are per beat in a midi file.
    @param verbose: If to print verbose output
    @param control_change_resolution: interval in which control change automation events are set between endpoints
    @param pitch_bend_resolution: interval in which control change automation events are set between endpoints
    """
    verbose_print(f'Merging {input_file} into {output_file}...', verbose=verbose)

    ardour_dom = xml.dom.minidom.parse(input_file)
    session_name = ardour_dom.getElementsByTagName('Session')[0].getAttribute('name')
    sample_rate = int(ardour_dom.getElementsByTagName('Session')[0].getAttribute('sample-rate'))
    verbose_print(f'Found session name "{session_name}"', verbose=verbose)

    session_dir = os.path.dirname(input_file)
    midi_dir = os.path.join(session_dir, 'interchange', session_name, 'midifiles')

    # Create an output midi file
    output_midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    # ardour_route_id_to_midi = {}

    # Create a meta tracks for markers (e.g. events that indicate a loop) 
    if create_meta_track:
        meta_tack = midi_create_meta_track(ardour_dom, output_midi, ticks_per_beat, sample_rate, verbose=verbose)

    messages = defaultdict(list)

    for route in ardour_dom.getElementsByTagName("Route"):
        if route.getAttribute("default-type") == "midi":
            route_name = route.getAttribute('name')
            # track = output_midi.add_track(name=route_name)
            route_id = route.getAttribute('id')
            # ardour_route_id_to_midi[route_id] = track

            # Insturment name
            messages[route_id].append(mido.MetaMessage("instrument_name", name=route_name))
            
            # Tempo
            tempo = ardour_dom.getElementsByTagName('Tempo')[0]
            bpm = int(tempo.getAttribute('beats-per-minute'))
            messages[route_id].append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))

            # Meter
            meter = ardour_dom.getElementsByTagName('Meter')[0]
            meter_numerator = int(meter.getAttribute('divisions-per-bar'))
            meter_denominator = int(meter.getAttribute('note-type'))
            messages[route_id].append(mido.MetaMessage('time_signature', numerator=meter_numerator, denominator=meter_denominator))

            controllables = list(route.getElementsByTagName('Controllable'))
            # Initial program change
            program_change_controllables = [c for c in controllables if c.getAttribute('name') == 'midi-pgm-change-0']
            if len(program_change_controllables) > 0:
                program = int(program_change_controllables[0].getAttribute('value'))
                messages[route_id].append(mido.Message('program_change', program=program))
            else:
                program = None
            
            # Initial pitch bender
            for c in controllables:
                if m := re.match(r"midi-pitch-bender-([0-9]+)", c.getAttribute('name')):
                    channel = int(m.group(1))
                    if channel == 0:
                        messages[route_id].append(mido.Message('pitchwheel', channel=0, time=0, pitch=int(c.getAttribute('value')) - 8192))


            # Initial midi control changes
            for c in controllables:
                # Format is midicc-{channel}-{value}
                if m := re.match(r"midicc-([0-9]+)-([0-9]+)", c.getAttribute('name')):
                    channel, control = int(m.group(1)), int(m.group(2))
                    if channel == 0:
                        messages[route_id].append(mido.Message('control_change', channel=channel, control=control, value=int(c.getAttribute('value'))))

    midi_sources, automation_interpolation_styles = get_sources(ardour_dom, midi_dir, verbose=verbose)
    assemble(ardour_dom, output_midi, midi_sources, messages, verbose=verbose, ticks_per_beat=args.ticks_per_beat)
    messages = sanitize_messages(messages)
    for route_id, msgs in messages.items():
        track = output_midi.add_track()
        track_add_messages(track, msgs, automation_interpolation_styles=automation_interpolation_styles[route_id], verbose=verbose,
            control_change_resolution=control_change_resolution, pitch_bend_resolution=pitch_bend_resolution)

    output_midi.save(output_file)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_project', help='Path of the input ardour project.')
    parser.add_argument('output_file', help='Path to the merged output midi filename.') 
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', 
        help='If verbose output will be printed.')
    parser.add_argument('-m', '--meta', dest='create_meta_track', action='store_true', 
        help='Create a meta track for tempo, position markers, etc.')
    parser.add_argument('--ticks-per-beat', dest='ticks_per_beat', default=19200, type=int,
         help='How many ticks there are per beat. Defaults to ardour`s 19200')
    parser.add_argument('-c', '--control-change-resolution', dest='control_change_resolution', default=4, type=int, 
        help='Resolution for linarizing midi control change events. Sets the value interval in which events are added between two endpoints.')
    parser.add_argument('-p', '--pitch-bend-resolution', dest='pitch_bend_resolution', default=256, type=int, 
        help='Resolution for linarizing pitch bend events. Sets the value interval in which events are added between two endpoints.')
    args = parser.parse_args()
    export(args.input_project, args.output_file, create_meta_track=args.create_meta_track, 
        verbose=args.verbose, control_change_resolution=args.control_change_resolution, pitch_bend_resolution=args.pitch_bend_resolution)