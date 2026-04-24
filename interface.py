import streamlit as st
import os
import json
import pandas as pd
from moviepy import VideoFileClip

from utils.downloader import download_yt
from utils.transcriber import transcribe
from utils.highlights import get_viral_highlights
from utils.image_gen import generate_broll_images
from utils.visualizer import assemble_final_video, create_fast_preview
from app import check_environment
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

# Helper to ensure we have a single float
def ensure_float(val):
    if isinstance(val, list):
        return float(val[0]) # Take the first item if it's a list
    return float(val)
    # if isinstance(val, list):
    #     return float(val[0])  # Take the first item if it's a list
    # # Remove non-numeric characters and convert to float
    # cleaned_val = ''.join(filter(str.isdigit, val))
    # return float(cleaned_val)
    
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
if 'preview_path' not in st.session_state:
    st.session_state.preview_path = None
if 'preview_paths' not in st.session_state:
    st.session_state.preview_paths = {}
if 'broll_durations' not in st.session_state:
    st.session_state.broll_durations = {}
if 'broll_paths' not in st.session_state:
    st.session_state.broll_paths = []
if 'preview_with_broll_path' not in st.session_state:
    st.session_state.preview_with_broll_path = None
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.title("⚙️ AI Director")
    disabled_input = st.session_state.plan is not None or st.session_state.analysis_running
    vid_input = st.text_input("Youtube Link:", st.session_state.video_path, disabled=disabled_input)
    
     # If clicked, set a session lock and run the long job under the spinner.
    if st.button("Step 1: AI Analysis", type="primary", disabled=disabled_input):
        st.session_state.analysis_running = True
        st.rerun() # Rerun to trigger the spinner and run the analysis in the main area

    if st.session_state.analysis_running:
        with st.spinner("🎬 Analyzing video..."):
            try:
                # 1. Download & Transcribe
                video_file = download_yt(vid_input)
                st.session_state.video_path = video_file
                full_transcript = transcribe(video_file)

                # 2. Get 3 highlight alternatives
                  = get_viral_highlights(full_transcript)
                st.session_state.highlights = highlights_list
                st.session_state.selected_highlight_idx = 0
                
                # 3. Generate previews for all 3 highlights
                st.session_state.preview_paths = {}
                for idx, highlight in enumerate(highlights_list):
                    start_t = ensure_float(highlight['start'])
                    end_t = ensure_float(highlight['end'])
                    
                    # Filter the transcript to this highlight's segment
                    viral_transcript = [
                        item for item in full_transcript 
                        if ensure_float(item['start']) >= start_t and ensure_float(item['end']) <= end_t
                    ]
                    
                    # Generate preview with a unique output file for each highlight
                    preview_path = f"output/preview_highlight_{idx}.mp4"
                    preview = create_fast_preview(st.session_state.video_path, start_t, end_t, viral_transcript, output_path=preview_path)
                    if preview and os.path.exists(preview):
                        st.session_state.preview_paths[idx] = preview
                        
                # Set the plan to the selected highlight (default is first one)
                st.session_state.plan = st.session_state.highlights[st.session_state.selected_highlight_idx]
                
                # Set transcript and preview for the selected highlight
                start_t = ensure_float(st.session_state.plan['start'])
                end_t = ensure_float(st.session_state.plan['end'])
                viral_transcript = [
                    item for item in full_transcript 
                    if ensure_float(item['start']) >= start_t and ensure_float(item['end']) <= end_t
                ]
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
    st.subheader("🎬 Select Your Highlight (3 AI Alternatives)")
    
    tab1, tab2, tab3 = st.tabs(["Option 1", "Option 2", "Option 3"])
    
    for tab_idx, (tab, highlight) in enumerate(zip([tab1, tab2, tab3], st.session_state.highlights)):
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
                    start_t = ensure_float(st.session_state.plan['start'])
                    end_t = ensure_float(st.session_state.plan['end'])
                    
                    from utils.downloader import download_yt
                    from utils.transcriber import transcribe
                    full_transcript = transcribe(st.session_state.video_path)
                    
                    viral_transcript = [
                        item for item in full_transcript 
                        if ensure_float(item['start']) >= start_t and ensure_float(item['end']) <= end_t
                    ]
                    st.session_state.transcript = viral_transcript
                    st.session_state.preview_path = st.session_state.preview_paths.get(tab_idx)
                    st.rerun()
            
            # Show the preview video
            if tab_idx in st.session_state.preview_paths:
                st.video(st.session_state.preview_paths[tab_idx])
            else:
                st.warning("Preview not available")
    
    st.divider()
    
    # --- TOP ACTION BAR ---
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.subheader(f"🎬 Highlight: {st.session_state.plan.get('reason', 'Viral Clip')}")
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
            
            # Regenerate preview with edited transcript
            preview = create_fast_preview(st.session_state.video_path, start_t, end_t, edited_transcript)
            if preview and os.path.exists(preview):
                st.session_state.preview_path = preview
                st.rerun()
            else:
                st.error("Preview refresh failed")

    with col_right:
        st.markdown("### 📺 Video Preview")
        if st.session_state.preview_path:
            st.video(st.session_state.preview_path)
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
    st.subheader("📸 AI B-Roll Prompts")
    st.info("Edit prompts, adjust duration, and generate preview-ready B-roll images.")
    
    updated_broll = []
    # Use the cues from the AI plan
    broll_cues = st.session_state.plan.get('broll', [])

    if 'broll_durations' not in st.session_state:
        st.session_state.broll_durations = {i: 3.0 for i in range(len(broll_cues))}

    # We create a 2-column grid so the boxes are nice and wide
    for i in range(0, len(broll_cues), 2):
        row_cols = st.columns(2)
        
        # Left Prompt Card
        with row_cols[0]:
            cue = broll_cues[i]
            st.markdown(f"**Visual #{i+1}**")
            n_time = st.number_input(f"Start Time (s)", value=ensure_float(cue['time']), key=f"time_{i}")
            n_duration = st.slider(f"Duration (s)", 1.0, 10.0, st.session_state.broll_durations.get(i, 3.0), 0.5, key=f"duration_{i}")
            n_prompt = st.text_area(f"Image Prompt", value=cue['prompt'], height=100, key=f"prompt_{i}")
            st.session_state.broll_durations[i] = n_duration
            updated_broll.append({"time": n_time, "prompt": n_prompt, "duration": n_duration})
            
        # Right Prompt Card (if there is an even number of cues)
        if i + 1 < len(broll_cues):
            with row_cols[1]:
                cue = broll_cues[i+1]
                st.markdown(f"**Visual #{i+2}**")
                n_time = st.number_input(f"Start Time (s)", value=ensure_float(cue['time']), key=f"time_{i+1}")
                n_duration = st.slider(f"Duration (s)", 1.0, 10.0, st.session_state.broll_durations.get(i+1, 3.0), 0.5, key=f"duration_{i+1}")
                n_prompt = st.text_area(f"Image Prompt", value=cue['prompt'], height=100, key=f"prompt_{i+1}")
                st.session_state.broll_durations[i+1] = n_duration
                updated_broll.append({"time": n_time, "prompt": n_prompt, "duration": n_duration})

    if st.button("✨ Generate & Preview B-Roll", use_container_width=True):
        with st.status("🎨 Generating B-roll and preview...", expanded=True) as status:
            broll_paths = []
            for i, cue in enumerate(updated_broll):
                image_filename = f"output/broll_{st.session_state.selected_highlight_idx}_{i}.png"
                status.write(f"Creating Image {i+1}: {cue['prompt'][:40]}...")
                path = generate_broll_images(cue['prompt'], image_filename)
                broll_paths.append(path)
            st.session_state.broll_paths = broll_paths
            st.session_state.plan['broll'] = updated_broll
            preview_file = f"output/preview_with_broll_{st.session_state.selected_highlight_idx}.mp4"
            preview = assemble_final_video(
                st.session_state.video_path,
                st.session_state.plan,
                broll_paths,
                transcript_data=edited_transcript,
                output_path=preview_file
            )
            if preview and os.path.exists(preview):
                st.session_state.preview_with_broll_path = preview
                st.session_state.preview_path = preview
                st.success("Preview updated with B-roll.")
            else:
                st.error("Failed to create B-roll preview.")

    # --- RENDER EXECUTION ---
    if render_ready:
        # 1. Update the session plan with the LATEST UI values
        st.session_state.plan['start'] = clip_range[0]
        st.session_state.plan['end'] = clip_range[1]
        st.session_state.plan['broll'] = updated_broll
        # Use the edited transcript text for captions
        current_transcript = edited_transcript

        with st.status("🚀 Processing your Short...", expanded=True) as status:
            # Clear old images from memory/list
            broll_paths = []
            
            # 2. Generate AI Visuals based on YOUR edited prompts
            st.write("🎨 Generating AI B-Roll images...")
            for i, cue in enumerate(updated_broll):
                image_filename = f"output/broll_{st.session_state.selected_highlight_idx}_{i}.png"
                status.write(f"Creating Image {i+1}: {cue['prompt'][:40]}...")
                
                # Call your image_gen logic
                path = generate_broll_images(cue['prompt'], image_filename)
                broll_paths.append(path)
            
            # 3. Assemble Final Video
            st.write("🎬 Combining video, captions, and B-roll...")
            # Note: We pass the edited transcript so captions are 100% correct
            final_video_path = assemble_final_video(
                st.session_state.video_path, 
                st.session_state.plan, 
                broll_paths,
                transcript_data=current_transcript 
            )
            
            status.update(label="✅ Video Ready!", state="complete", expanded=False)
        
        # 4. Display the result
        st.divider()
        st.balloons()
        st.success("Your AI Short is ready for download!")
        st.video(final_video_path)
        
        with open(final_video_path, "rb") as file:
            st.download_button("📥 Download Short", file, file_name="ai_short.mp4")

else:
    st.info("👈 Enter a YouTube URL in the sidebar to get started.")