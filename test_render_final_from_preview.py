import argparse
import os
from utils.visualizer import assemble_final_video


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render final video from a preview highlight using the original source video and highlight timing."
    )
    parser.add_argument(
        "--source-video",
        required=True,
        help="Path to the original source video file used to generate the preview highlight.",
    )
    parser.add_argument(
        "--start",
        required=True,
        type=float,
        help="Start time of the highlight in seconds.",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=float,
        help="End time of the highlight in seconds.",
    )
    parser.add_argument(
        "--output",
        default="output/final_from_preview.mp4",
        help="Path for the final rendered output video.",
    )
    parser.add_argument(
        "--broll",
        nargs="*",
        default=[],
        help="Optional list of B-roll image paths to include in the final render.",
    )
    parser.add_argument(
        "--render-mode",
        default="tiktok",
        choices=["tiktok", "clip"],
        help="Render mode for the final video.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    plan = {
        "start": args.start,
        "end": args.end,
        "broll": [],
    }

    # If B-roll images are provided, assign them sequentially to the highlight.
    for idx, img_path in enumerate(args.broll):
        plan["broll"].append({
            "time": args.start + idx * 3.0,
            "duration": 3.0,
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Rendering final video from preview highlight:")
    print(f"  source video: {args.source_video}")
    print(f"  highlight range: {args.start}-{args.end}")
    print(f"  render mode: {args.render_mode}")
    if args.broll:
        print(f"  b-roll images: {args.broll}")
    print(f"  output path: {args.output}")

    result = assemble_final_video(
        args.source_video,
        plan,
        args.broll,
        transcript_data=None,
        output_path=args.output,
        render_mode=args.render_mode,
    )

    print(f"Final render complete: {result}")


if __name__ == "__main__":
    main()
