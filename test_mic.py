import sounddevice as sd
import numpy as np

def print_volume(indata, frames, time, status):
    if status:
        print(status)
    # indata is a numpy array of shape (frames, channels)
    # compute the peak amplitude
    peak = np.abs(indata).max()
    # print a visual bar
    bar = "=" * int(peak * 100)
    print(f"[{peak:0.3f}] {bar}")

print("Listening to microphone... Speak now to test volume levels (Ctrl+C to stop).")
print("If the peak value never goes above 0.5 when you talk normally,")
print("you need to lower the 'threshold' in configs/fast.yaml.")

with sd.InputStream(callback=print_volume):
    sd.sleep(100000)
