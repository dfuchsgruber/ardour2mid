#!/usr/bin/env python3

# Mostly based on https://github.com/dbolton/ArdourMIDIExport/blob/master/main.py

import os
import argparse
import typing
import warnings

import xml.dom.minidom
import mido
import numpy as np

ROUTE_ID_META_TRACK = -1

def verbose_print(*args, verbose : bool = False, **kwargs):
    if verbose:
        print(*args, **kwargs)

def get_sources(ardour_dom : xml.dom.minidom.Document, midi_dir : str) -> typing.Dict[str, str]:
    """ Gets all midi sources references by a ardour project
    @param ardour_dom: The ardour xml project tree
    @param midi_dir: Where to find the source midi files.
    @return: mapping from source ids to source names
    """
    sources = {}
    for source in ardour_dom.getElementsByTagName('Source'):
        if source.getAttribute('type') == 'midi':
            path = os.path.join(midi_dir, source.getAttribute('name'))
            if not os.path.exists(path):
                warnings.warn(f'Could not find midi source {path}', RuntimeWarning)
            else:
                sources[source.getAttribute('id')] = path
    return sources

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
            print(start_ticks)
            markers.append(name)
            marker_times.append(start_ticks)

    idx_sorted = np.argsort(marker_times)
    marker_times, markers = np.array(marker_times)[idx_sorted], np.array(markers)[idx_sorted]
    marker_times[1:] -= marker_times[:-1] # Calculate differences between marker positions

    for time, marker in zip(marker_times, markers):
        print(time, marker)
        msg = mido.MetaMessage('marker', time=time, text=marker)
        meta_track.append(msg)

    output_midi.print_tracks()

    return meta_track

def assemble(ardour_dom : xml.dom.minidom.Document, output_midi : mido.MidiFile, midi_sources : typing.Dict[str, str],
                ardour_route_id_to_midi : typing.Dict[str, mido.MidiTrack], ticks_per_beat=19200, verbose : bool = False):
    for playlist_idx, playlist in enumerate(ardour_dom.getElementsByTagName('Playlist')): # Each playlist will be one track
        verbose_print(f'Iterating playlist {playlist_idx}...', verbose=verbose)
        if playlist.getAttribute('type') == 'midi':
            route_id = playlist.getAttribute('orig-track-id')
            track = ardour_route_id_to_midi[route_id]
            verbose_print(f'\tMatched to route {route_id}', verbose=verbose)
            
            # Iterate through all the regions
            total_time = 0
            previous_region_end_time = 0
            for region_idx, region in enumerate(playlist.getElementsByTagName('Region')):
                verbose_print(f'\tRegion {region_idx}:', verbose=verbose)
                source_midi_path = midi_sources[region.getAttribute('source-0')]
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
                                track.append(msg_to_insert)
                                num_events_inserted += 1
                            else:
                                verbose_print(f'\t\tIgnored midi event at {msg.time}', verbose=verbose)
                        else:
                            # Turn off all nodes that are still playing after the region ended
                            for note in range(128):
                                if note_is_on[note]:
                                    track.append(mido.Message('note_off', note=note))
                                    note_is_on[note] = False
                                    verbose_print(f'\t\tNote {note} was playing at end of region. Insert note_off event.', verbose=verbose)
                
                verbose_print(f'\t\tInserted {num_events_inserted} midi events.', verbose=verbose)
                previous_region_end_time = total_time


def export(input_file : str, output_file : str, create_meta_track : bool = True, ticks_per_beat : int = 19200, verbose : bool = False):
    """ Merges and exports the midi files of a project. 
    @param input_file: Input adrour project file
    @param output_file: Output midi file
    @param create_meta_track: If to create a meta track for loops, tempo etc.
    @param ticks_per_beat: How many ticks there are per beat in a midi file.
    @verbose: If to print verbose output
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
    ardour_route_id_to_midi = {}

    # Create a master 
    if create_meta_track:
        ardour_route_id_to_midi[ROUTE_ID_META_TRACK] = midi_create_meta_track(ardour_dom, output_midi, ticks_per_beat, sample_rate, verbose=verbose)

    for route in ardour_dom.getElementsByTagName("Route"):
        if route.getAttribute("default-type") == "midi":
            route_name = route.getAttribute('name')
            track = output_midi.add_track(name=route_name)
            route_id = route.getAttribute('id')
            ardour_route_id_to_midi[route_id] = track
            verbose_print(f'Created midi track "{route_name}" based on route id {route_id}', verbose=verbose)

            # Insturment name
            track.append(mido.MetaMessage("instrument_name", name=route_name))
            
            # Tempo
            tempo = ardour_dom.getElementsByTagName('Tempo')[0]
            bpm = int(tempo.getAttribute('beats-per-minute'))
            track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))

            # Meter
            meter = ardour_dom.getElementsByTagName('Meter')[0]
            meter_numerator = int(meter.getAttribute('divisions-per-bar'))
            meter_denominator = int(meter.getAttribute('note-type'))
            track.append(mido.MetaMessage('time_signature', numerator=meter_numerator, denominator=meter_denominator))


    midi_sources = get_sources(ardour_dom, midi_dir)
    assemble(ardour_dom, output_midi, midi_sources, ardour_route_id_to_midi, verbose=verbose, ticks_per_beat=args.ticks_per_beat)

    output_midi.save(output_file)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_project', help='Path of the input ardour project.')
    parser.add_argument('output_file', help='Path to the merged output midi filename.') 
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', help='If verbose output will be printed.')
    parser.add_argument('-m', '--meta', dest='create_meta_track', action='store_true', help='Create a meta track for tempo, position markers, etc.')
    parser.add_argument('--ticks-per-beat', dest='ticks_per_beat', default=19200, type=int)
    args = parser.parse_args()
    export(args.input_project, args.output_file, create_meta_track=args.create_meta_track, verbose=args.verbose)