from utils.visualizer import create_fast_preview

# Test with your actual data
result = create_fast_preview(
    "Supergirl Trailer Breakdown - The Comic The Cast  My Honest Take.mp4",
    0,
    10,
    [{"start": 0, "end": 5, "text": "test is working so far"}, {"start": 5, "end": 10, "text": "second part of the test"}],
    "output/test_preview.mp4"
)
print(f"Result: {result}")