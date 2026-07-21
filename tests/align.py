import sys, os
import numpy as np
from scipy.interpolate import interp1d
import music21 as m21
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import tactus_align as ta
from config import SCORE_PATH, AUDIO_PATH

def test_align(mei_path: str, audio_path: str):
    '''
    Main synchronization function.

    :param mei_path: path to MEI file
    :param audio_path: path to audio file
    '''
    print("Loading audio...", end="")
    X, sr = ta.load_audio(audio_path, trim_threshold=30)
    print(f"Done: {X.shape}, sr={sr}")
    print("Loading score...", end="")
    score = ta.load_score(mei_path)
    print("Done.")
    print("Synthesizing score...", end="")
    Y, _ = ta.synthesize_score(score, sr)
    print(f"Done: {Y.shape}, sr={sr}")

    pipeline = ta.AlignmentPipeline(
        ta.ChromaCENS(win_len_smooth=5, n_fft=2048, hop_length=512),
        ta.BasicDTW(),
        metric='euclidean'
    )
    
    print("Running alignment...", end="")
    result = pipeline.run(X, Y, sr)
    print("Done.")

    print("Warping score to audio time...", end="")
    path_X_seconds, path_Y_seconds = ta.path_to_time_map(result.alignment.path, result.hop_length, sr)

    unique_indices = np.unique(path_Y_seconds, return_index=True)[1]
    time_map_Y_to_X = interp1d(
        path_Y_seconds[unique_indices], 
        path_X_seconds[unique_indices], 
        kind='linear', 
        fill_value="extrapolate"
    )
    print("Done.")

    part = score.parts[0] if score.parts else score
    
    # Flatten the stream so element offsets are global (relative to the start)
    flat_part = part.flatten()
    s_map = flat_part.secondsMap
    
    # Extract unique global Quarter Length (QL) offsets and their time in seconds
    ql_to_sec = {}
    for entry in s_map:
        global_ql = float(entry['element'].offset)
        if global_ql not in ql_to_sec:
            ql_to_sec[global_ql] = float(entry['offsetSeconds'])
            
    # Add the final end time so we can safely interpolate the end of the last measure
    if s_map:
        last_entry = s_map[-1]
        last_el = last_entry['element']
        end_ql = float(last_el.offset + last_el.quarterLength)
        end_sec = float(last_entry['offsetSeconds'] + last_entry['durationSeconds'])
        ql_to_sec[end_ql] = end_sec

    ql_offsets = sorted(ql_to_sec.keys())
    sec_offsets = [ql_to_sec[k] for k in ql_offsets]
    
    # Fallback to constant 120 BPM if secondsMap is empty
    if not ql_offsets:
        ql_to_sec_Y = lambda ql: float(ql * 0.5)
    else:
        ql_to_sec_Y = interp1d(
            ql_offsets, 
            sec_offsets, 
            kind='linear', 
            fill_value="extrapolate"
        )

    print("Building MEASURES dictionary...", end="")

    # ---------------------------------------------------------
    # 3. Build the MEASURES Dictionary (Using MEI XML IDs)
    # ---------------------------------------------------------
    import xml.etree.ElementTree as ET
    
    # Parse the raw MEI to get the exact XML elements
    tree = ET.parse(mei_path)
    ns = {
        'mei': 'http://www.music-encoding.org/ns/mei', 
        'xml': 'http://www.w3.org/XML/1998/namespace'
    }
    
    mei_measures = tree.findall('.//mei:measure', namespaces=ns)
    m21_measures = list(part.getElementsByClass(m21.stream.Measure))
    
    MEASURES = {}
    
    # Zip them together so we get the MEI ID and the music21 timing simultaneously
    for mei_meas, m21_meas in zip(mei_measures, m21_measures):
        
        # 1. Get the ID from the MEI element
        # In ElementTree, the 'xml:id' attribute is accessed using the expanded namespace URI
        m_id = mei_meas.get(f"{{{ns['xml']}}}id")
        
        # Fallback just in case the file formats it differently (e.g., 'id' or 'n')
        if not m_id:
            m_id = mei_meas.get('xml:id', f"m-{mei_meas.get('n', m21_meas.number)}")
            
        # 2. Get the timing from the music21 element
        start_ql = m21_meas.offset
        end_ql = m21_meas.offset + m21_meas.quarterLength
        
        # 3. Convert and Warp
        start_time_unwarped = ql_to_sec_Y(start_ql)
        end_time_unwarped = ql_to_sec_Y(end_ql)
        
        start_time_warped = float(time_map_Y_to_X(start_time_unwarped))
        end_time_warped = float(time_map_Y_to_X(end_time_unwarped))
        
        MEASURES[m_id] = (max(0.0, start_time_warped), max(0.0, end_time_warped))

    print("Done.")

    return MEASURES

if __name__ == "__main__":    
    measures = test_align(SCORE_PATH, AUDIO_PATH)
    for m_id, (start, end) in measures.items():
        print(f"Measure {m_id}: Start = {start:.3f}s, End = {end:.3f}s")