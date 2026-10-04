import cv2

class VideoRecorder:
    def __init__(self, output_path="../outputs/output_video.mp4", fps=20, frame_size=(640, 480)):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.frame_size = frame_size

    def write(self, frame):
        # Ensure frame matches expected size (webcam resolutions can vary)
        resized = cv2.resize(frame, self.frame_size)
        self.writer.write(resized)

    def add_subtitle(self, frame, text, sub_height=60):
        """Adds a black subtitle bar with the given text at the bottom of the frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - sub_height), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Simple text wrapping so long sentences don't get cut off
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        words = text.split(" ")
        line = ""
        y = h - sub_height + 25
        for word in words:
            test_line = f"{line} {word}".strip()
            (tw, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if tw > w - 20:
                cv2.putText(frame, line, (10, y), font, font_scale, (255, 255, 255), thickness)
                line = word
                y += 22
            else:
                line = test_line
        cv2.putText(frame, line, (10, y), font, font_scale, (255, 255, 255), thickness)
        return frame

    def release(self):
        self.writer.release()