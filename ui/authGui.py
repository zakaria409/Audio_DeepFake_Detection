import pyaudio
import wave
import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import time
import os
import numpy as np
from keras.models import load_model
import model
from model import extract_features,predict_genre

class VoiceRecorderApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Voice Recorder")
        self.master.geometry("540x420")
        self.master.configure(bg='lightblue')
        
        self.output_file = "recorded_audio.wav"
        self.is_recording = False
        self.selected_file_path = None
        
        self.create_widgets()
        self.model = load_model('speakRec2.h5')

    def create_widgets(self):
        self.record_button = tk.Button(self.master, text="Record", command=self.toggle_record, height=3, width=6)
        self.record_button.pack()
        self.label = tk.Label(self.master, text="00:00:00")
        self.label.pack()

        self.select_button = tk.Button(self.master, text="Select Audio File", command=self.select_audio_file)
        self.select_button.pack(pady=10)

        self.text_display0 = tk.Text(self.master, height=3, width=10)
        self.text_display0.pack(pady=10)
        self.text_display0.insert("1.0", "\nresult\n")
        self.center_text(self.text_display0)

        self.update_button = tk.Button(self.master, text="Classify", command=self.update_text)
        self.update_button.pack()

        self.text_display1 = tk.Text(self.master, height=3, width=20)
        self.text_display1.pack(pady=10, side="left", padx=(5, 0))
        self.text_display1.insert("1.0", "\nresult\n")
        self.center_text(self.text_display1)

        self.select_text = tk.Text(self.master, height=3, width=20)
        self.select_text.pack(pady=10, side="left", padx=(5, 0))

        self.identify_button = tk.Button(self.master, text="Identify", command=self.identify)
        self.identify_button.pack(pady=10, side="left", padx=(5, 0))

    def toggle_record(self):
        if not self.is_recording:
            self.is_recording = True
            self.record_button.config(text="Recording", fg="red")
            threading.Thread(target=self.record_audio).start()
        else:
            self.record_button.config(text="Record", fg="black")
            self.is_recording = False

    def record_audio(self, duration=5, sample_rate=44100, channels=2, chunk=1024):
        audio_format = pyaudio.paInt16
        audio = pyaudio.PyAudio()
        
        stream = audio.open(format=audio_format, channels=channels,
                            rate=sample_rate, input=True,
                            frames_per_buffer=chunk)
        
        start = time.time()
        frames = []

        while self.is_recording:
            data = stream.read(chunk)
            frames.append(data)

            passed = time.time() - start
            mins, secs = divmod(passed, 60)
            hrs, mins = divmod(mins, 60)
            self.master.after(0, self.label.config, {"text": f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"})
        
        print("Recording finished.")
        
        stream.stop_stream()
        stream.close()
        audio.terminate()

        i = 1
        while os.path.exists(f"recording{i}.wav"):
            i += 1

        sound_file = wave.open(f"recording{i}.wav", "wb")
        sound_file.setnchannels(channels)
        sound_file.setsampwidth(audio.get_sample_size(audio_format))
        sound_file.setframerate(sample_rate)
        sound_file.writeframes(b"".join(frames))
        sound_file.close()

    def select_audio_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Wave files", "*.wav"), ("MP3 files", "*.mp3")])
        if file_path:
            self.selected_file_path = file_path
            file_name = os.path.basename(file_path)
            self.select_text.delete("1.0", "end")
            self.select_text.insert("1.0", file_name)
            print("Selected audio file:", file_path)

    def update_text(self):
        if not self.selected_file_path:
            messagebox.showerror("Error", "No file selected")
            return

        genre_mapping = {0: "spoofed", 1: "Real"}
        features = extract_features(self.selected_file_path)
        predicted_genre = predict_genre(self.model, features, genre_mapping)

        self.text_display0.delete("1.0", "end")
        self.text_display0.insert("1.0", predicted_genre)
        self.center_text(self.text_display0)

    def identify(self):
        if not self.selected_file_path:
            messagebox.showerror("Error", "No file selected")
            return

        speaker_folders = ['Nelson_Mandela', 'Magaret_Tarcher', 'Benjamin_Netanyau', 'Jens_Stoltenberg', 'Julia_Gillard', 'Zakaria_Sameh']
        features = extract_features(self.selected_file_path)
        y_pred = self.model.predict(features)
        y_pred = np.argmax(y_pred, axis=-1)[0]

        self.text_display1.delete("1.0", "end")
        self.text_display1.insert("1.0", speaker_folders[int(y_pred)])
        self.center_text(self.text_display1)

    def center_text(self, text_display):
        text_display.tag_configure("center", justify='center')
        text_display.tag_add("center", "1.0", "end")

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceRecorderApp(root)
    root.mainloop()