import json
import sys
import matplotlib.pyplot as plt

def count_people_per_frame(json_path, plot=False):
    with open(json_path, 'r') as f:
        data = json.load(f)
    frame_indices = []
    people_counts = []
    for frame in data:
        frame_idx = frame.get("frame_index", None)
        people = frame.get("people", [])
        print(f"Frame {frame_idx}: {len(people)} people detected")
        frame_indices.append(frame_idx)
        people_counts.append(len(people))
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(frame_indices, people_counts, marker='o')
        plt.xlabel('Frame Index')
        plt.ylabel('Number of People Detected')
        plt.title('People Detected Per Frame')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python people_count.py <input_json> [--plot]")
        sys.exit(1)
    plot = len(sys.argv) == 3 and sys.argv[2] == "--plot"
    count_people_per_frame(sys.argv[1], plot=plot)