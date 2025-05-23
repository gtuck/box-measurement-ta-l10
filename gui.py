import PySimpleGUI as sg
import cv2
import threading
import os
import numpy as np
import json 
from measureBox import setup_camera, get_intensity_image, \
                       calibrate_intrinsics as measurebox_calibrate_intrinsics, \
                       calibrate_depth_scale as measurebox_calibrate_depth_scale, \
                       measure_box_dimensions as measurebox_measure_box_dimensions

# --- Global State Variables ---
camera = None
intrinsics_active = False
depth_calib_active = False
start_sampling_signal = threading.Event()
measurement_active = False
full_run_active = False
active_thread = None 
active_thread_type = None 

# --- Logging Helper ---
def log_message(message, window, log_key='-LOG-'):
    if window and message is not None: 
        window[log_key].print(str(message))

# --- Parameter Validation Helper for GUI Thread ---
def validate_intrinsics_params(values, window):
    try:
        if not all([values['-INTRINSICS_FILE-'], values['-CB_DIMS-'], values['-CB_SIZE-'], values['-CB_FRAMES-']]):
            log_message("Error: All intrinsic calibration parameters must be provided.", window)
            return None
        dims = tuple(map(int, values['-CB_DIMS-'].split(',')))
        if len(dims) != 2: raise ValueError("Checkerboard dimensions must be two comma-separated integers (e.g., 7,6).")
        square_size = float(values['-CB_SIZE-'])
        if square_size <= 0: raise ValueError("Square size must be positive.")
        num_frames = int(values['-CB_FRAMES-'])
        if num_frames <= 0: raise ValueError("Number of frames must be positive.")
        return dims, square_size, num_frames, values['-INTRINSICS_FILE-']
    except ValueError as e:
        log_message(f"Error: Invalid intrinsic calibration parameter: {e}", window)
        return None

def validate_depth_scale_params(values, window):
    try:
        if not all([values['-DEPTH_SCALE_FILE-'], values['-KNOWN_DIST-'], values['-DS_SAMPLES-']]):
            log_message("Error: All depth scale calibration parameters must be provided.", window)
            return None
        known_distance = float(values['-KNOWN_DIST-'])
        if known_distance <= 0: raise ValueError("Known distance must be positive.")
        num_samples = int(values['-DS_SAMPLES-'])
        if num_samples <= 0: raise ValueError("Number of samples must be positive.")
        return known_distance, num_samples, values['-DEPTH_SCALE_FILE-']
    except ValueError as e:
        log_message(f"Error: Invalid depth scale calibration parameter: {e}", window)
        return None

def validate_measurement_params(values, window):
    try:
        plane_thresh = float(values['-PLANE_THRESH-'])
        cluster_eps = float(values['-CLUSTER_EPS-'])
        cluster_min_points = int(values['-CLUSTER_MIN_POINTS-'])
        # Bounds checks can be added here if necessary, e.g. plane_thresh > 0
        return plane_thresh, cluster_eps, cluster_min_points
    except ValueError as e:
        log_message(f"Error: Invalid measurement parameter: {e}", window)
        return None

# --- Thread Functions (largely unchanged, parameter validation was already there but now it's duplicated for early GUI feedback) ---
def run_intrinsic_calibration_thread(window, output_file, dims, square_size, num_frames):
    global intrinsics_active, active_thread, active_thread_type
    log_message_thread = lambda msg: window.write_event_value('-LOG_FROM_THREAD-', msg)
    update_frame_count_thread = lambda current, total: window.write_event_value('-UPDATE_INTRINSICS_FRAME_COUNT-', (current, total))
    try:
        update_frame_count_thread(0, num_frames) 
        log_message_thread(f"Starting intrinsic calibration: Output='{output_file}', Dims={dims}, Size={square_size}, Frames={num_frames}")
        mtx, dist, messages = measurebox_calibrate_intrinsics(
            camera_obj=camera, output_file=output_file, checkerboard_dims=dims,
            square_size=square_size, num_frames=num_frames, gui_window=window, 
            image_key='-IMAGE-', log_callback=log_message_thread)
        window.write_event_value('-INTRINSICS_RESULT-', (mtx, dist, messages))
    except Exception as e: # Catch-all for unexpected errors in measureBox or thread logic
        log_message_thread(f"Unhandled exception in intrinsic calibration thread: {str(e)}")
        window.write_event_value('-INTRINSICS_RESULT-', (None, None, [f"Thread exception: {str(e)}"]))
    finally:
        intrinsics_active = False
        if active_thread_type == 'intrinsics': active_thread_type = None; active_thread = None

def run_depth_calibration_thread(window, output_file, known_distance, num_samples, start_event):
    global depth_calib_active, active_thread, active_thread_type
    log_message_thread = lambda msg: window.write_event_value('-LOG_FROM_THREAD-', msg)
    try:
        log_message_thread(f"Starting depth scale calibration: Output='{output_file}', KnownDist={known_distance}, Samples={num_samples}")
        scale_factor, messages = measurebox_calibrate_depth_scale(
            camera_obj=camera, output_file=output_file, known_distance=known_distance,
            num_samples=num_samples, gui_window=window, image_key='-IMAGE-',
            start_event=start_event, log_callback=log_message_thread)
        window.write_event_value('-DEPTH_CALIB_RESULT-', (scale_factor, messages))
    except Exception as e:
        log_message_thread(f"Unhandled exception in depth calibration thread: {str(e)}")
        window.write_event_value('-DEPTH_CALIB_RESULT-', (None, [f"Thread exception: {str(e)}"]))
    finally:
        depth_calib_active = False
        if active_thread_type == 'depth': active_thread_type = None; active_thread = None

def run_measurement_thread(window, intrinsics_file, depth_scale_file, plane_thresh, cluster_eps, cluster_min_points):
    global measurement_active, active_thread, active_thread_type
    log_message_thread = lambda msg: window.write_event_value('-LOG_FROM_THREAD-', msg)
    try:
        log_message_thread(f"Starting measurement: Intrinsics='{intrinsics_file}', DepthScale='{depth_scale_file}'")
        dimensions_dict, messages = measurebox_measure_box_dimensions(
            camera_obj=camera, intrinsics_file_path=intrinsics_file, depth_scale_file_path=depth_scale_file,
            plane_dist_thresh=plane_thresh, cluster_eps=cluster_eps, cluster_min_points=cluster_min_points,
            log_callback=log_message_thread, gui_window=window, image_key='-IMAGE-')
        window.write_event_value('-MEASUREMENT_RESULT-', (dimensions_dict, messages))
    except Exception as e:
        log_message_thread(f"Unhandled exception in measurement thread: {str(e)}")
        window.write_event_value('-MEASUREMENT_RESULT-', (None, [f"Thread exception: {str(e)}"]))
    finally:
        measurement_active = False
        if active_thread_type == 'measure': active_thread_type = None; active_thread = None

def run_full_run_thread(window, gui_values): # Validation remains inside for Full Run due to complexity
    global full_run_active, active_thread, active_thread_type
    all_messages = []
    log_message_thread = lambda msg: window.write_event_value('-LOG_FROM_THREAD-', msg)
    def _log_full_run(message): log_message_thread(message); all_messages.append(str(message))
    try:
        _log_full_run("Full Run Started.")
        active_thread_type = 'full_run'
        # Parameter parsing and validation for all steps (as previously implemented)
        intrinsics_output_file = gui_values['-INTRINSICS_FILE-']; dims_str = gui_values['-CB_DIMS-']; square_size_str = gui_values['-CB_SIZE-']; num_frames_str = gui_values['-CB_FRAMES-']
        depth_scale_output_file = gui_values['-DEPTH_SCALE_FILE-']; known_dist_str = gui_values['-KNOWN_DIST-']; num_samples_str = gui_values['-DS_SAMPLES-']
        plane_thresh_str = gui_values['-PLANE_THRESH-']; cluster_eps_str = gui_values['-CLUSTER_EPS-']; cluster_min_points_str = gui_values['-CLUSTER_MIN_POINTS-']

        try: # Intrinsics params
            if not all([intrinsics_output_file, dims_str, square_size_str, num_frames_str]): raise ValueError("Missing intrinsic parameters.")
            dims = tuple(map(int, dims_str.split(','))); assert len(dims) == 2, "Checkerboard Dims: 2 values."
            square_size = float(square_size_str); assert square_size > 0, "Square Size > 0."
            num_frames = int(num_frames_str); assert num_frames > 0, "Num Frames > 0."
        except Exception as e: _log_full_run(f"Full Run Error: Invalid Intrinsics params: {e}"); window.write_event_value('-FULL_RUN_RESULT-', {'success': False, 'messages': all_messages}); return
        try: # Depth params
            if not all([depth_scale_output_file, known_dist_str, num_samples_str]): raise ValueError("Missing depth scale parameters.")
            known_distance = float(known_dist_str); assert known_distance > 0, "Known Distance > 0."
            num_samples = int(num_samples_str); assert num_samples > 0, "Num Samples > 0."
        except Exception as e: _log_full_run(f"Full Run Error: Invalid Depth Scale params: {e}"); window.write_event_value('-FULL_RUN_RESULT-', {'success': False, 'messages': all_messages}); return
        try: # Measurement params
            plane_thresh = float(plane_thresh_str); cluster_eps = float(cluster_eps_str); cluster_min_points = int(cluster_min_points_str)
        except Exception as e: _log_full_run(f"Full Run Error: Invalid Measurement params: {e}"); window.write_event_value('-FULL_RUN_RESULT-', {'success': False, 'messages': all_messages}); return

        _log_full_run("Full Run - Step 1: Intrinsic Calibration...")
        mtx, dist, intr_msgs = measurebox_calibrate_intrinsics(camera, intrinsics_output_file, dims, square_size, num_frames, window, '-IMAGE-', _log_full_run)
        if mtx is None: _log_full_run("Full Run - Step 1: Intrinsic Calibration FAILED."); window.write_event_value('-FULL_RUN_RESULT-', {'success': False, 'messages': all_messages}); return
        _log_full_run("Full Run - Step 1: Intrinsic Calibration SUCCESSFUL.")

        _log_full_run("Full Run - Step 2: Depth Scale Calibration...")
        depth_start_event = threading.Event(); depth_start_event.set()
        scale_factor, ds_msgs = measurebox_calibrate_depth_scale(camera, depth_scale_output_file, known_distance, num_samples, window, '-IMAGE-', depth_start_event, _log_full_run)
        if scale_factor is None: _log_full_run("Full Run - Step 2: Depth Scale Calibration FAILED."); window.write_event_value('-FULL_RUN_RESULT-', {'success': False, 'messages': all_messages}); return
        _log_full_run("Full Run - Step 2: Depth Scale Calibration SUCCESSFUL.")

        _log_full_run("Full Run - Step 3: Box Measurement...")
        dimensions_dict, meas_msgs = measurebox_measure_box_dimensions(camera, intrinsics_output_file, depth_scale_output_file, plane_thresh, cluster_eps, cluster_min_points, _log_full_run, window, '-IMAGE-')
        if dimensions_dict is None: _log_full_run("Full Run - Step 3: Box Measurement FAILED."); window.write_event_value('-FULL_RUN_RESULT-', {'success': False, 'messages': all_messages, 'status_message': 'Measurement failed.'}); return
        _log_full_run("Full Run - Step 3: Box Measurement SUCCESSFUL.")
        
        _log_full_run("Full Run Completed Successfully!")
        window.write_event_value('-FULL_RUN_RESULT-', {'success': True, 'dimensions': dimensions_dict, 'messages': all_messages})
    except Exception as e:
        _log_full_run(f"Unhandled exception in Full Run thread: {str(e)}")
        all_messages.append(f"Full Run Thread exception: {str(e)}")
        window.write_event_value('-FULL_RUN_RESULT-', {'success': False, 'dimensions': None, 'messages': all_messages})
    finally:
        full_run_active = False
        if active_thread_type == 'full_run': active_thread_type = None; active_thread = None

# --- Define the layout ---
sg.theme('Reddit')
camera_col = [[sg.Button('Connect Camera', key='-CONNECT_CAMERA-'), sg.Text("Camera: Not Connected", key='-CAMERA_STATUS-', size=(30,1))],[sg.Text('Camera Port:'), sg.InputText(key='-PORT-', default_text='scan', size=(15,1))]]
intrinsics_tab_layout = [[sg.Text('Intrinsics File:'), sg.InputText(key='-INTRINSICS_FILE-', default_text='camera_intrinsics.npz', size=(30,1))],[sg.Text('Checkerboard Dims (cols,rows):'), sg.InputText(key='-CB_DIMS-', default_text='7,6', size=(10,1))],[sg.Text('Square Size (m):'), sg.InputText(key='-CB_SIZE-', default_text='0.025', size=(10,1))],[sg.Text('Num Frames:'), sg.InputText(key='-CB_FRAMES-', default_text='20', size=(5,1))],[sg.Text(f"Collected: 0/0", key='-INTRINSICS_FRAME_COUNT_TEXT-', size=(20,1))] ]
depth_scale_tab_layout = [[sg.Text('Depth Scale File:'), sg.InputText(key='-DEPTH_SCALE_FILE-', default_text='depth_scale.json', size=(30,1))],[sg.Text('Known Distance (m):'), sg.InputText(key='-KNOWN_DIST-', default_text='1.0', size=(10,1))],[sg.Text('Num Samples:'), sg.InputText(key='-DS_SAMPLES-', default_text='50', size=(5,1))],[sg.Button("Start Sampling", key='-START_SAMPLING-', disabled=True)]]
measurement_tab_layout = [[sg.Text('Plane Threshold (m):'), sg.InputText(key='-PLANE_THRESH-', default_text='0.01', size=(10,1))],[sg.Text('Cluster Epsilon (m):'), sg.InputText(key='-CLUSTER_EPS-', default_text='0.02', size=(10,1))],[sg.Text('Cluster Min Points:'), sg.InputText(key='-CLUSTER_MIN_POINTS-', default_text='100', size=(5,1))]]
param_tab_group = sg.TabGroup([[sg.Tab('Intrinsics Calib.', intrinsics_tab_layout, key='-TAB_INTRINSICS-')],[sg.Tab('Depth Scale Calib.', depth_scale_tab_layout, key='-TAB_DEPTH_SCALE-')],[sg.Tab('Measurement Params', measurement_tab_layout, key='-TAB_MEASUREMENT-')]])
action_buttons_col = [[sg.Button("Calibrate Intrinsics", key='-CALIB_INTRINSICS-', disabled=True),sg.Button("Calibrate Depth Scale", key='-CALIB_DEPTH-', disabled=True),sg.Button("Measure Box", key='-MEASURE_BOX-', disabled=True),sg.Button("Full Run", key='-FULL_RUN-', disabled=True)]]
layout = [[sg.Column(camera_col)],[sg.HSeparator()],[sg.Column(action_buttons_col)],[sg.HSeparator()],[sg.Column([[param_tab_group]]),sg.Image(key='-IMAGE-', size=(640, 480), background_color='black')],[sg.Text("Log Output:")],[sg.Multiline(key='-LOG-', size=(100, 20), autoscroll=True, reroute_stdout=False, reroute_stderr=False, font='monospace 8', do_not_clear=True)],[sg.Button('Quit', key='-QUIT-')]]
window = sg.Window('Box Measurement TA-L10 GUI', layout, finalize=True)

def update_buttons_for_process_status(is_starting_process):
    # Disable all main action buttons if a process is starting or active
    # Enable them if no process is active AND camera is connected
    disable_all = is_starting_process or (active_thread and active_thread.is_alive())
    enable_main = not disable_all and camera is not None

    window['-CALIB_INTRINSICS-'].update(disabled=disable_all or not camera)
    window['-CALIB_DEPTH-'].update(disabled=disable_all or not camera)
    window['-MEASURE_BOX-'].update(disabled=disable_all or not camera)
    window['-FULL_RUN-'].update(disabled=disable_all or not camera)
    
    if not is_starting_process or active_thread_type != 'depth': # Keep start sampling disabled unless depth calib specifically enables it
        window['-START_SAMPLING-'].update(disabled=True)


# --- Main Event Loop ---
while True:
    event, values = window.read(timeout=100) 

    if event == sg.WIN_CLOSED or event == '-QUIT-':
        if active_thread and active_thread.is_alive():
            log_message("Quit signaled. Attempting to stop active thread...", window)
            if intrinsics_active: intrinsics_active = False
            if depth_calib_active: depth_calib_active = False; start_sampling_signal.set()
            if measurement_active: measurement_active = False
            if full_run_active: full_run_active = False
            active_thread.join(timeout=2) 
            if active_thread.is_alive(): log_message(f"Active {active_thread_type or 'thread'} did not stop in time.", window)
        break

    if event == '-CONNECT_CAMERA-': 
        if camera: 
            log_message("Disconnecting camera...", window)
            if active_thread and active_thread.is_alive(): 
                if intrinsics_active: intrinsics_active = False
                if depth_calib_active: depth_calib_active = False; start_sampling_signal.set()
                if measurement_active: measurement_active = False
                if full_run_active: full_run_active = False
                active_thread.join(timeout=1)
                log_message(f"Active {active_thread_type or 'process'} stopped due to disconnect.", window)
            try: camera.close()
            except Exception as e: log_message(f"Error disconnecting camera: {e}", window)
            camera = None; active_thread = None; active_thread_type = None
            window['-CAMERA_STATUS-'].update("Camera: Not Connected")
            window['-CONNECT_CAMERA-'].update("Connect Camera")
            update_buttons_for_process_status(False) # Update based on camera disconnect
            intrinsics_active = depth_calib_active = measurement_active = full_run_active = False
        else: 
            port_val = values['-PORT-'].strip()
            port_to_use = None if port_val.lower() == 'scan' or not port_val else port_val
            log_message(f"Attempting to connect to camera on port: {'scan' if port_to_use is None else port_to_use}", window)
            window.refresh()
            cam_obj, msgs = setup_camera(port_to_use)
            for msg in msgs: log_message(msg, window)
            if cam_obj:
                camera = cam_obj; cam_info = camera.info()
                window['-CAMERA_STATUS-'].update(f"Camera: Connected ({cam_info.model} on {cam_info.port})")
                window['-CONNECT_CAMERA-'].update("Disconnect Camera")
            else:
                window['-CAMERA_STATUS-'].update("Camera: Connection Failed"); camera = None
            update_buttons_for_process_status(False) # Update based on camera connect/fail
    
    elif event == '-CALIB_INTRINSICS-':
        if not camera: log_message("Error: Camera not connected.", window); continue
        if active_thread and active_thread.is_alive(): log_message(f"Error: Process '{active_thread_type}' is active.", window); continue
        
        params = validate_intrinsics_params(values, window)
        if not params: continue # Validation failed, message already logged
        dims, square_size, num_frames, output_file = params

        intrinsics_active = True; active_thread_type = 'intrinsics'
        update_buttons_for_process_status(True)
        log_message("Starting intrinsic calibration thread...", window)
        active_thread = threading.Thread(target=run_intrinsic_calibration_thread, args=(window, output_file, dims, square_size, num_frames), daemon=True)
        active_thread.start()

    elif event == '-CALIB_DEPTH-':
        if not camera: log_message("Error: Camera not connected.", window); continue
        if active_thread and active_thread.is_alive(): log_message(f"Error: Process '{active_thread_type}' is active.", window); continue
        
        params = validate_depth_scale_params(values, window)
        if not params: continue
        known_distance, num_samples, output_file = params

        depth_calib_active = True; active_thread_type = 'depth'
        start_sampling_signal.clear() 
        update_buttons_for_process_status(True)
        window['-START_SAMPLING-'].update(disabled=False) # Specifically enable this
        log_message("Starting depth scale calibration thread... Press 'Start Sampling' when ready.", window)
        active_thread = threading.Thread(target=run_depth_calibration_thread, args=(window, output_file, known_distance, num_samples, start_sampling_signal), daemon=True)
        active_thread.start()

    elif event == '-MEASURE_BOX-':
        if not camera: log_message("Error: Camera not connected.", window); continue
        if active_thread and active_thread.is_alive(): log_message(f"Error: Process '{active_thread_type}' is active.", window); continue
        
        intr_file = values['-INTRINSICS_FILE-']; ds_file = values['-DEPTH_SCALE_FILE-']
        if not os.path.exists(intr_file): log_message(f"Error: Intrinsics file '{intr_file}' not found.", window); continue
        if not os.path.exists(ds_file): log_message(f"Error: Depth scale file '{ds_file}' not found.", window); continue
        
        meas_params = validate_measurement_params(values, window)
        if not meas_params: continue
        plane_thresh, cluster_eps, cluster_min_points = meas_params
            
        measurement_active = True; active_thread_type = 'measure'
        update_buttons_for_process_status(True)
        log_message("Starting box measurement thread...", window)
        active_thread = threading.Thread(target=run_measurement_thread, args=(window, intr_file, ds_file, plane_thresh, cluster_eps, cluster_min_points), daemon=True)
        active_thread.start()

    elif event == '-FULL_RUN-': # Validation for Full Run remains within its thread due to complexity
        if not camera: log_message("Error: Camera not connected.", window); continue
        if active_thread and active_thread.is_alive(): log_message(f"Error: Process '{active_thread_type}' is active.", window); continue
        full_run_active = True; active_thread_type = 'full_run'
        update_buttons_for_process_status(True)
        log_message("Starting Full Run thread...", window)
        active_thread = threading.Thread(target=run_full_run_thread, args=(window, values), daemon=True)
        active_thread.start()

    elif event == '-START_SAMPLING-':
        if depth_calib_active:
            start_sampling_signal.set(); log_message("Sampling start signal sent.", window)
            window['-START_SAMPLING-'].update(disabled=True) 
        else: log_message("Start Sampling clicked, but depth calibration is not active.", window)
            
    elif event == '-LOG_FROM_THREAD-': log_message(values[event], window)
    elif event == '-UPDATE_INTRINSICS_FRAME_COUNT-': window['-INTRINSICS_FRAME_COUNT_TEXT-'].update(f"Collected: {values[event][0]}/{values[event][1]}")

    elif event == '-INTRINSICS_RESULT-':
        mtx, dist, result_messages = values[event]
        for msg in result_messages: log_message(msg, window)
        if mtx is not None: log_message("Intrinsic calibration successful!", window)
        else: log_message("Intrinsic calibration failed.", window)
        intrinsics_active = False
        if active_thread_type == 'intrinsics': active_thread = None; active_thread_type = None
        update_buttons_for_process_status(False)

    elif event == '-DEPTH_CALIB_RESULT-':
        scale_factor, result_messages = values[event]
        for msg in result_messages: log_message(msg, window)
        if scale_factor is not None: log_message(f"Depth scale calibration successful! Scale: {scale_factor:.5f}", window)
        else: log_message("Depth scale calibration failed.", window)
        depth_calib_active = False
        if active_thread_type == 'depth': active_thread = None; active_thread_type = None
        update_buttons_for_process_status(False) # This will also disable -START_SAMPLING-
            
    elif event == '-MEASUREMENT_RESULT-':
        dimensions_dict, result_messages = values[event]
        for msg in result_messages: log_message(msg, window)
        if dimensions_dict:
            log_message("Box measurement successful!", window)
            log_message(f"  Length (mm): {dimensions_dict['length_mm']:.1f}", window)
            log_message(f"  Width (mm): {dimensions_dict['width_mm']:.1f}", window)
            log_message(f"  Height (mm): {dimensions_dict['height_mm']:.1f}", window)
            log_message(f"  Center (m): X={dimensions_dict['center_x_m']:.3f}, Y={dimensions_dict['center_y_m']:.3f}, Z={dimensions_dict['center_z_m']:.3f}", window)
        else:
            log_message("Box measurement failed or no box found.", window)
        measurement_active = False
        if active_thread_type == 'measure': active_thread = None; active_thread_type = None
        update_buttons_for_process_status(False)

    elif event == '-FULL_RUN_RESULT-':
        result_data = values[event]
        for msg in result_data.get('messages', []): log_message(msg, window)
        if result_data.get('success'):
            log_message("Full Run COMPLETED SUCCESSFULLY!", window)
            dims = result_data.get('dimensions')
            if dims: log_message(f"Measured Dimensions: L={dims['length_mm']:.1f}, W={dims['width_mm']:.1f}, H={dims['height_mm']:.1f} mm", window)
        else: log_message(result_data.get('status_message', 'Full Run FAILED at some stage.'), window)
        full_run_active = False
        if active_thread_type == 'full_run': active_thread = None; active_thread_type = None
        update_buttons_for_process_status(False)


if camera: 
    try: camera.close(); print("Camera closed on exit.")
    except Exception as e: print(f"Error closing camera on exit: {e}")
if window: window.close()
print("GUI Closed.")
