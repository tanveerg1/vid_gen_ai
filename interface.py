import os
os.environ.setdefault("STREAMLIT_SERVER_MAX_UPLOAD_SIZE", "1024")
import streamlit as st
import json
import pandas as pd
import importlib
from moviepy.video.io.VideoFileClip import VideoFileClip

from utils.downloader import download_yt
from utils.transcriber import transcribe
from utils.highlights import get_viral_highlights
from utils.image_gen import generate_broll_images
from utils import visualizer
from utils.visualizer import assemble_final_video, create_fast_preview
from app import check_environment

# Force reload the visualizer module to pick up latest changes
importlib.reload(visualizer)
# from streamlit_video_editor import video_editor_timeline
from streamlit_video_editor import st_video_editor

st.set_page_config(page_title="Vid Gen UI", layout="wide")

@st.cache_resource
def initialize_environment():
    """Run environment checks only once per session."""
    check_environment()
    return True

# Initialize environment on first load
initialize_environment()

# Helper to convert timestamps to seconds

def parse_time_to_seconds(val):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, list) and val:
        return parse_time_to_seconds(val[0])
    if not isinstance(val, str):
        return float(val or 0.0)

    s = val.strip()
    if not s:
        return 0.0
    if s.endswith('s'):
        s = s[:-1].strip()

    if ':' in s:
        parts = [float(p) if p else 0.0 for p in s.split(':')]
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours = 0.0
            minutes, seconds = parts
        else:
            return float(parts[0])
        return hours * 3600.0 + minutes * 60.0 + seconds

    try:
        return float(s)
    except ValueError:
        cleaned = ''.join(ch for ch in s if ch.isdigit() or ch == '.' or ch == '-')
        return float(cleaned or 0.0)


def ensure_float(val):
    return parse_time_to_seconds(val)


def sanitize_filename(filename):
    filename = os.path.basename(filename)
    filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).strip()
    return filename.replace(' ', '_') or 'uploaded_video'


def validate_local_video_path(path):
    if not path:
        return None
    if not os.path.exists(path):
        return None
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb"):
            pass
    except PermissionError:
        return "permission_error"
    except Exception:
        return None
    return path
    
# Initialize Session State    
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'highlights' not in st.session_state:
    st.session_state.highlights = None
if 'selected_highlight_idx' not in st.session_state:
    st.session_state.selected_highlight_idx = 0
if 'plan' not in st.session_state:
    st.session_state.plan = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'youtube_url' not in st.session_state:
    st.session_state.youtube_url = ""
if 'uploaded_video_path' not in st.session_state:
    st.session_state.uploaded_video_path = None
if 'local_video_path' not in st.session_state:
    st.session_state.local_video_path = ""
if 'preview_path' not in st.session_state:
    st.session_state.preview_path = None
if 'preview_paths' not in st.session_state:
    st.session_state.preview_paths = {}
if 'broll_durations' not in st.session_state:
    st.session_state.broll_durations = {}
if 'broll_paths' not in st.session_state:
    st.session_state.broll_paths = []
if 'broll_model_id' not in st.session_state:
    st.session_state.broll_model_id = os.environ.get("BROLL_MODEL_ID", "dreamlike-art/dreamlike-photoreal-2.0")
if 'broll_model_custom' not in st.session_state:
    st.session_state.broll_model_custom = ""
if 'preview_with_broll_path' not in st.session_state:
    st.session_state.preview_with_broll_path = None
if 'render_mode' not in st.session_state:
    st.session_state.render_mode = "TikTok"
if 'include_broll' not in st.session_state:
    st.session_state.include_broll = True
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.title("⚙️ AI Director")
    disabled_input = st.session_state.plan is not None or st.session_state.analysis_running
    vid_input = st.text_input("YouTube Link:", st.session_state.youtube_url, disabled=disabled_input)
    st.session_state.local_video_path = st.text_input(
        "Or enter a local video file path",
        st.session_state.local_video_path,
        placeholder=r"C:\path\to\video.mp4",
        disabled=disabled_input,
        help="For large local files, enter the full path to avoid Streamlit upload limits.",
    )
    uploaded_file = st.file_uploader(
        "Or upload a local video file",
        type=['mp4', 'mov', 'avi', 'mkv'],
        disabled=disabled_input,
    )

    local_path_state = validate_local_video_path(st.session_state.local_video_path)

    if uploaded_file is not None:
        upload_dir = os.path.join("output", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        upload_name = sanitize_filename(uploaded_file.name)
        upload_path = os.path.join(upload_dir, upload_name)
        if not os.path.exists(upload_path):
            with open(upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_video_path = upload_path
        st.session_state.video_path = upload_path
    elif local_path_state == "permission_error":
        st.error("Cannot read the local file. Please check file permissions or choose a different file.")
    elif local_path_state:
        st.session_state.video_path = local_path_state
    elif st.session_state.local_video_path:
        st.warning("Local path not found or unreadable. Please check the path or use a YouTube URL/upload.")
    elif st.session_state.uploaded_video_path:
        st.session_state.video_path = st.session_state.uploaded_video_path

    st.markdown("---")
    st.subheader("🎯Output Type")
    st.session_state.render_mode = st.radio(
        "Choose the type of clip to create",
        ["TikTok", "Clip"],
        index=0 if st.session_state.render_mode == "TikTok" else 1,
        horizontal=True,
        disabled=disabled_input,
    )
    if st.session_state.render_mode == "Clip":
        st.session_state.include_broll = False

    if st.session_state.render_mode == "TikTok":
        st.markdown("---")
        st.subheader("🎨 B-roll Model")
        model_options = [
            "dreamlike-art/dreamlike-photoreal-2.0",
            "stabilityai/sdxl-turbo",
            "runwayml/stable-diffusion-v1-5",
            "hakurei/waifu-diffusion",
            "Custom model ID"
        ]
        try:
            default_index = model_options.index(st.session_state.broll_model_id)
        except ValueError:
            default_index = len(model_options) - 1

        selected_model = st.selectbox(
            "Choose a B-roll model",
            model_options,
            index=default_index,
            disabled=disabled_input,
        )
        if selected_model == "Custom model ID":
            st.session_state.broll_model_custom = st.text_input(
                "Custom model ID",
                value=st.session_state.broll_model_custom,
                placeholder="e.g. user/repo-model-name",
                disabled=disabled_input,
            )
            if st.session_state.broll_model_custom:
                st.session_state.broll_model_id = st.session_state.broll_model_custom
        else:
            st.session_state.broll_model_id = selected_model

    # If clicked, set a session lock and run the long job under the spinner.
    if st.button("Step 1: AI Analysis", type="primary", disabled=disabled_input):
        st.session_state.analysis_running = True
        st.rerun() # Rerun to trigger the spinner and run the analysis in the main area

    if st.session_state.analysis_running:
        with st.spinner("🎬 Analyzing video..."):
            try:
                # 1. Use the user-provided source: local path > uploaded file > YouTube
                local_path_state = validate_local_video_path(st.session_state.local_video_path)
                if local_path_state:
                    if local_path_state == "permission_error":
                        raise PermissionError("Cannot read the local file. Please check file permissions.")
                    video_file = local_path_state
                elif uploaded_file is not None or st.session_state.uploaded_video_path:
                    video_file = st.session_state.uploaded_video_path
                else:
                    st.session_state.youtube_url = vid_input
                    video_file = download_yt(vid_input)
                st.session_state.video_path = video_file
                full_transcript = transcribe(video_file)

                # 2. Load source duration and get 3 highlight alternatives
                with VideoFileClip(video_file) as source_clip:
                    st.session_state.total_duration = source_clip.duration

                highlights_list  = get_viral_highlights(
                    full_transcript,
                    mode=st.session_state.render_mode.lower(),
                    video_duration=st.session_state.total_duration,
                )
                if highlights_list is None:
                    raise ValueError("Failed to parse AI highlights from Gemma 4")

                for highlight in highlights_list:
                    if 'start' not in highlight and 'start_time' in highlight:
                        highlight['start'] = highlight['start_time']
                    if 'end' not in highlight and 'end_time' in highlight:
                        highlight['end'] = highlight['end_time']
                    highlight['start'] = parse_time_to_seconds(highlight.get('start', highlight.get('start_time', 0.0)))
                    highlight['end'] = parse_time_to_seconds(highlight.get('end', highlight.get('end_time', 0.0)))
                    if st.session_state.render_mode == "clip":
                        highlight['broll'] = []
                    if 'broll' in highlight:
                        for cue in highlight['broll']:
                            if 'time' not in cue and 'cue_time' in cue:
                                cue['time'] = cue['cue_time']
                            cue['time'] = parse_time_to_seconds(cue.get('time', 0.0))
                st.session_state.highlights = highlights_list
                st.session_state.selected_highlight_idx = 0
                
                # 3. Generate previews for all 3 highlights
                st.session_state.preview_paths = {}
                for idx, highlight in enumerate(highlights_list):
                    start_t = ensure_float(highlight['start'])
                    end_t = ensure_float(highlight['end'])
                    
                    ## OLD WORKING CODE
                    # Filter the transcript to this highlight's segment
                    # viral_transcript = [
                        # item for item in full_transcript 
                        # if ensure_float(item['start']) >= start_t and ensure_float(item['end']) <= end_t
                    # ]
                    
                    ### NEW CODE
                    viral_transcript = []
                    for item in full_transcript:
                        if ensure_float(item['start']) >= start_t and ensure_float(item['end']) <= end_t:
                            # Deep copy the segment map to maintain underlying whisper timestamps
                            viral_transcript.append(item.copy())
                    #####
                    
                    # Generate preview with a unique output file for each highlight
                    preview_path = f"output/preview_highlight_{idx}.mp4"
                    preview = create_fast_preview(
                        st.session_state.video_path,
                        start_t,
                        end_t,
                        viral_transcript,
                        output_path=preview_path,
                        render_mode=st.session_state.render_mode.lower(),
                    )
                    if preview and os.path.exists(preview):
                        st.session_state.preview_paths[idx] = preview
                        
                # Set the plan to the selected highlight (default is first one)
                st.session_state.plan = st.session_state.highlights[st.session_state.selected_highlight_idx]
                
                # Set transcript and preview for the selected highlight
                start_t = ensure_float(st.session_state.plan['start'])
                end_t = ensure_float(st.session_state.plan['end'])
                
                ### OLD WORKING CODE
                # viral_transcript = [
                    # item for item in full_transcript 
                    # if ensure_float(item['start']) >= start_t and ensure_float(item['end']) <= end_t
                # ]
                
                ### NEW CODE
                viral_transcript = []
                for item in full_transcript:
                    if ensure_float(item['start']) >= start_t and ensure_float(item['end']) <= end_t:
                        viral_transcript.append(item.copy())
                ###
                st.session_state.transcript = viral_transcript
                st.session_state.preview_path = st.session_state.preview_paths.get(st.session_state.selected_highlight_idx)
                
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
            finally:
                st.session_state.analysis_running = False
                st.rerun() # Rerun to update the UI with results
        st.success("Analysis Complete!")

# --- MAIN AREA: HIGHLIGHT SELECTION ---
if st.session_state.plan:
    # Get total video duration for the slider
    if 'total_duration' not in st.session_state:
        with VideoFileClip(st.session_state.video_path) as clip:
            st.session_state.total_duration = clip.duration
    
    # --- HIGHLIGHT SELECTION TABS ---
    st.subheader(f"🎬 Select Your Highlight ({len(st.session_state.highlights)} AI Alternatives)")
    
    tab_labels = [f"Option {i+1}" for i in range(len(st.session_state.highlights))]
    tabs = st.tabs(tab_labels)
    
    for tab_idx, (tab, highlight) in enumerate(zip(tabs, st.session_state.highlights)):
        with tab:
            col_info, col_btn = st.columns([3, 1])
            
            with col_info:
                st.markdown(f"**Why it works:** {highlight.get('reason', 'Viral clip')}")
                st.markdown(f"⏱️ **Duration:** {ensure_float(highlight['start']):.1f}s - {ensure_float(highlight['end']):.1f}s ({ensure_float(highlight['end']) - ensure_float(highlight['start']):.0f}s)")
            
            with col_btn:
                is_selected = st.session_state.selected_highlight_idx == tab_idx
                btn_label = "✓ Selected" if is_selected else "Select"
                btn_type = "secondary" if is_selected else "primary"
                if st.button(btn_label, key=f"select_highlight_{tab_idx}", type=btn_type, use_container_width=True):
                    st.session_state.selected_highlight_idx = tab_idx
                    st.session_state.plan = st.session_state.highlights[tab_idx]
                    
                    # Update transcript and preview for this highlight
                    st.session_state.plan['start'] = parse_time_to_seconds(st.session_state.plan.get('start', 0.0))
                    st.session_state.plan['end'] = parse_time_to_seconds(st.session_state.plan.get('end', 0.0))
                    start_t = st.session_state.plan['start']
                    end_t = st.session_state.plan['end']
                    
                    from utils.downloader import download_yt
                    from utils.transcriber import transcribe
                    full_transcript = transcribe(st.session_state.video_path)
                    
                    ### OLD CODE
                    # viral_transcript = [
                        # item for item in full_transcript 
                        # if ensure_float(item['start']) >= start_t and ensure_float(item['end']) <= end_t
                    # ]
                    ####
                    
                    ### NEW CODE
                    # Safely isolate the segment maps along with their deep dictionary values
                    viral_transcript = []
                    for item in full_transcript:
                        if ensure_float(item['start']) >= start_t and ensure_float(item['end']) <= end_t:
                            viral_transcript.append(item.copy())
                    ###
                    
                    st.session_state.transcript = viral_transcript
                    st.session_state.preview_path = st.session_state.preview_paths.get(tab_idx)
                    st.session_state.preview_with_broll_path = None
                    st.session_state.broll_paths = []
                    st.rerun()
            
            # Show the preview video
            if tab_idx in st.session_state.preview_paths:
                st.video(st.session_state.preview_paths[tab_idx], width=360)
            else:
                st.warning("Preview not available")
    
    st.divider()
    
    # --- TOP ACTION BAR ---
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.subheader(f"🎬 Highlight: {st.session_state.plan.get('reason', 'Viral Clip')}")
        if st.session_state.render_mode == "Clip":
            st.info("Clip mode: same original aspect ratio, no burned-in captions, and no B-roll.")
        else:
            st.session_state.include_broll = st.checkbox(
                "Include AI B-roll for TikTok",
                value=st.session_state.include_broll,
                help="When unchecked, TikTok renders captions only and skips image generation.",
                key="include_broll_checkbox",
            )
    with col_b:
        render_ready = st.button("🚀 Render Final Short", type="primary", width='stretch')

    # --- THE DUAL PANEL VIEW ---
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        st.markdown("### 📝 Transcript Editor")
        # Show the transcript exactly like Opus
        edited_transcript = st.data_editor(
            st.session_state.transcript,
            column_config={"text": st.column_config.TextColumn("Dialogue", width="large")},
            width='stretch',
            height=500
        )
        
        # Add Refresh Preview button
        if st.button("🔄 Refresh Preview", use_container_width=True):
            start_t = ensure_float(st.session_state.plan['start'])
            end_t = ensure_float(st.session_state.plan['end'])
               
            ### NEW CODE
            # Reconstruct the precise word timestamps for the preview data
            from utils.transcriber import transcribe
            full_transcript = transcribe(st.session_state.video_path)
            original_slice = [
                item.copy() for item in full_transcript 
                if ensure_float(item['start']) >= start_t and ensure_float(item['end']) <= end_t
            ]
            
            preview_transcript = []
            if edited_transcript:
                for i, edited_item in enumerate(edited_transcript):
                    if i < len(original_slice):
                        matched_orig = original_slice[i]
                        if edited_item.get('text', '').strip() == matched_orig.get('text', '').strip():
                            preview_transcript.append(matched_orig)
                        else:
                            updated_item = matched_orig.copy()
                            updated_item['text'] = edited_item['text']
                            
                            words_text = edited_item['text'].split()
                            if words_text:
                                duration = updated_item['end'] - updated_item['start']
                                spacing = duration / len(words_text)
                                updated_item['words'] = [
                                    {
                                        'start': updated_item['start'] + (idx * spacing),
                                        'end': updated_item['start'] + ((idx + 1) * spacing),
                                        'text': txt
                                    }
                                    for idx, txt in enumerate(words_text)
                                ]
                            preview_transcript.append(updated_item)
                    else:
                        preview_transcript.append(edited_item)
            else:
                preview_transcript = original_slice
            ###
            
            # Regenerate the current preview, preserving the selected highlight filename
            if st.session_state.preview_with_broll_path:
                preview_path = st.session_state.preview_with_broll_path
                preview = assemble_final_video(
                    st.session_state.video_path,
                    st.session_state.plan,
                    st.session_state.broll_paths,
                    # transcript_data=edited_transcript,
                    transcript_data=preview_transcript,
                    output_path=preview_path,
                    render_mode=st.session_state.render_mode.lower(),
                )
            else:
                preview_path = st.session_state.preview_paths.get(
                    st.session_state.selected_highlight_idx,
                    f"output/preview_highlight_{st.session_state.selected_highlight_idx}.mp4",
                )
                preview = create_fast_preview(
                    st.session_state.video_path,
                    start_t,
                    end_t,
                    # edited_transcript,
                    preview_transcript,
                    output_path=preview_path,
                    render_mode=st.session_state.render_mode.lower(),
                )

            if preview and os.path.exists(preview):
                st.session_state.preview_path = preview
                if not st.session_state.preview_with_broll_path:
                    st.session_state.preview_paths[st.session_state.selected_highlight_idx] = preview
                st.rerun()
            else:
                st.error("Preview refresh failed")

    with col_right:
        st.markdown("### 📺 Video Preview")
        if st.session_state.preview_path:
            st.video(st.session_state.preview_path, width=360)
            # Timeline Range Slider right under the video
            clip_range = st.slider(
                "Adjust Clip Timing", 
                0.0, st.session_state.total_duration, 
                (ensure_float(st.session_state.plan['start']), ensure_float(st.session_state.plan['end']))
            )
        else:
            st.warning("Preview not available.")
    
    # --- B-ROLL PROMPT SECTION (Bottom) ---
    st.markdown("---")
    updated_broll = []

    if st.session_state.render_mode == "TikTok":
        st.subheader("📸 AI B-Roll Prompts")
        st.info("Edit prompts, adjust duration, and generate preview-ready B-roll images.")

        broll_cues = st.session_state.plan.get('broll', [])
        selected_idx = st.session_state.selected_highlight_idx
        if selected_idx not in st.session_state.broll_durations:
            st.session_state.broll_durations[selected_idx] = {i: 3.0 for i in range(len(broll_cues))}

        if st.session_state.include_broll and broll_cues:
            # We create a 2-column grid so the boxes are nice and wide
            for i in range(0, len(broll_cues), 2):
                row_cols = st.columns(2)

                # Left Prompt Card
                with row_cols[0]:
                    cue = broll_cues[i]
                    st.markdown(f"**Visual #{i+1}**")
                    n_time = st.number_input(f"Start Time (s)", value=ensure_float(cue['time']), key=f"time_{selected_idx}_{i}")
                    n_duration = st.slider(
                        f"Duration (s)", 1.0, 10.0,
                        st.session_state.broll_durations[selected_idx].get(i, 3.0),
                        0.5,
                        key=f"duration_{selected_idx}_{i}",
                    )
                    n_prompt = st.text_area(
                        f"Image Prompt",
                        value=cue['prompt'],
                        height=100,
                        key=f"prompt_{selected_idx}_{i}",
                    )
                    st.session_state.broll_durations[selected_idx][i] = n_duration
                    updated_broll.append({"time": n_time, "prompt": n_prompt, "duration": n_duration})

                # Right Prompt Card (if there is an even number of cues)
                if i + 1 < len(broll_cues):
                    with row_cols[1]:
                        cue = broll_cues[i+1]
                        st.markdown(f"**Visual #{i+2}**")
                        n_time = st.number_input(f"Start Time (s)", value=ensure_float(cue['time']), key=f"time_{selected_idx}_{i+1}")
                        n_duration = st.slider(
                            f"Duration (s)", 1.0, 10.0,
                            st.session_state.broll_durations[selected_idx].get(i+1, 3.0),
                            0.5,
                            key=f"duration_{selected_idx}_{i+1}",
                        )
                        n_prompt = st.text_area(
                            f"Image Prompt",
                            value=cue['prompt'],
                            height=100,
                            key=f"prompt_{selected_idx}_{i+1}",
                        )
                        st.session_state.broll_durations[selected_idx][i+1] = n_duration
                        updated_broll.append({"time": n_time, "prompt": n_prompt, "duration": n_duration})

            if st.button("✨ Generate & Preview B-Roll", use_container_width=True):
                with st.status("🎨 Generating B-roll and preview...", expanded=True) as status:
                    broll_paths = []
                    for i, cue in enumerate(updated_broll):
                        image_filename = f"output/broll_{st.session_state.selected_highlight_idx}_{i}.png"
                        status.write(f"Creating Image {i+1}: {cue['prompt'][:40]}...")
                        path = generate_broll_images(cue['prompt'], image_filename, model_id=st.session_state.broll_model_id)
                        broll_paths.append(path)
                    st.session_state.broll_paths = broll_paths
                    st.session_state.plan['broll'] = updated_broll
                    preview_file = f"output/preview_with_broll_{st.session_state.selected_highlight_idx}.mp4"
                    preview = assemble_final_video(
                        st.session_state.video_path,
                        st.session_state.plan,
                        broll_paths,
                        transcript_data=edited_transcript,
                        output_path=preview_file,
                        render_mode=st.session_state.render_mode.lower(),
                    )
                    if preview and os.path.exists(preview):
                        st.session_state.preview_with_broll_path = preview
                        st.session_state.preview_path = preview
                        st.success("Preview updated with B-roll.")
                    else:
                        st.error("Failed to create B-roll preview.")
        else:
            st.info("B-roll generation is disabled for this TikTok output. Final video will render captions only.")
    else:
        st.subheader("🎬 Clip mode")
        st.info("Clip output preserves the original video aspect ratio, skips burned-in captions, and does not expose B-roll controls.")

    # --- RENDER EXECUTION ---

    # --- RENDER EXECUTION ---
    if render_ready:
        # 1. Update the session plan with the LATEST UI values
        st.session_state.plan['start'] = clip_range[0]
        st.session_state.plan['end'] = clip_range[1]
        st.session_state.plan['broll'] = updated_broll if st.session_state.render_mode == "TikTok" and st.session_state.include_broll else []
        # Use the edited transcript text for captions only in TikTok mode
        current_transcript = edited_transcript if st.session_state.render_mode == "TikTok" else None

        with st.status("🚀 Processing your Short...", expanded=True) as status:
            broll_paths = []

            if st.session_state.render_mode == "TikTok" and st.session_state.include_broll and updated_broll:
                st.write("🎨 Generating AI B-roll images...")
                for i, cue in enumerate(updated_broll):
                    image_filename = f"output/broll_{st.session_state.selected_highlight_idx}_{i}.png"
                    status.write(f"Creating Image {i+1}: {cue['prompt'][:40]}...")
                    path = generate_broll_images(cue['prompt'], image_filename, model_id=st.session_state.broll_model_id)
                    broll_paths.append(path)
            else:
                status.write("Skipping B-roll generation for this render.")

            # 3. Assemble Final Video
            if st.session_state.render_mode == "TikTok":
                st.write("🎬 Combining video, captions, and B-roll...")
            else:
                st.write("🎬 Rendering clip with original aspect ratio and no burned-in captions...")

            final_video_path = assemble_final_video(
                st.session_state.video_path,
                st.session_state.plan,
                broll_paths,
                transcript_data=current_transcript,
                output_path=f"output/final_{st.session_state.render_mode.lower()}_{st.session_state.selected_highlight_idx}.mp4",
                render_mode=st.session_state.render_mode.lower(),
            )
            
            status.update(label="✅ Video Ready!", state="complete", expanded=False)
        
        # 4. Display the result
        st.divider()
        st.balloons()
        st.success("Your AI Short is ready for download!")
        st.video(final_video_path, width=360)
        
        with open(final_video_path, "rb") as file:
            st.download_button("📥 Download Short", file, file_name="ai_short.mp4")

else:
    st.info("👈 Enter a YouTube URL or upload a local video in the sidebar to get started.")