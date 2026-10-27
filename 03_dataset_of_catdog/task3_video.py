"""Task 3 视频部分：手动裁脸、五种处理效果、mp4v 输出。
"""

# ==================== 板块 1：路径与裁脸 ====================
import os
import cv2
import numpy as np

project_folder = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(project_folder, "output")
os.makedirs(output_folder, exist_ok=True)

image_path = input("请输入猫或狗图片的完整路径（不加引号）：").strip()
image_path = os.path.expanduser(image_path)
image = cv2.imread(image_path)
if image is None:
    raise RuntimeError("图片读取失败，请检查路径")

height = image.shape[0]
width = image.shape[1]
# 缩小大图片以便在屏幕上框选，小图片不放大。
scale = min(1.0, 900 / width, 600 / height)
preview_width = max(1, int(width * scale))
preview_height = max(1, int(height * scale))
preview = cv2.resize(image, (preview_width, preview_height))
print("请用鼠标框选脸部，按 Enter 确认；按 C 取消")
x, y, w, h = cv2.selectROI("Select Face", preview, True, False)
cv2.destroyAllWindows()
if w == 0 or h == 0:
    raise RuntimeError("没有选择有效的脸部区域")

# 按实际预览宽高换算坐标，避免缩放取整带来的误差。
x1 = int(x * width / preview_width)
y1 = int(y * height / preview_height)
x2 = int((x + w) * width / preview_width)
y2 = int((y + h) * height / preview_height)
# 数组切片：先 y（行），再 x（列）。
face = image[y1:y2, x1:x2].copy()
face_path = os.path.join(output_folder, "cropped_face.png")
if not cv2.imwrite(face_path, face):
    raise RuntimeError("裁剪图片保存失败")

# ==================== 板块 2：打开源视频 ====================
video_path = input("请输入源视频完整路径（不加引号）：").strip()
video_path = os.path.expanduser(video_path)
output_video_path = os.path.join(output_folder, "processed_video.mp4")
if os.path.realpath(video_path) == os.path.realpath(output_video_path):
    raise ValueError("源视频不能是即将写入的 processed_video.mp4")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    cap.release()
    raise RuntimeError("视频打开失败，请检查路径")
fps = cap.get(cv2.CAP_PROP_FPS)
if not np.isfinite(fps) or fps <= 0:
    cap.release()
    raise RuntimeError("视频帧率无效")
success, frame = cap.read()
if not success:
    cap.release()
    raise RuntimeError("无法读取视频第一帧")

# 输出采用偶数尺寸；奇数尺寸只裁掉最下面或最右边的一个像素。
video_height = frame.shape[0] - frame.shape[0] % 2
video_width = frame.shape[1] - frame.shape[1] % 2
if video_width < 2 or video_height < 2:
    cap.release()
    raise RuntimeError("视频尺寸无效")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    output_video_path, fourcc, fps, (video_width, video_height)
)
if not writer.isOpened():
    cap.release()
    writer.release()
    raise RuntimeError("无法创建输出视频，请检查 mp4v 编码支持")

# ==================== 板块 3：调整头像 ====================
resize_scale = min(
    video_width * 0.3 / face.shape[1],
    video_height * 0.3 / face.shape[0],
)
new_width = max(1, int(face.shape[1] * resize_scale))
new_height = max(1, int(face.shape[0] * resize_scale))
small_face = cv2.resize(face, (new_width, new_height))
left = min(20, video_width - new_width)
top = min(20, video_height - new_height)
frame_count_value = cap.get(cv2.CAP_PROP_FRAME_COUNT)
total_frames = 0
if np.isfinite(frame_count_value) and frame_count_value > 0:
    total_frames = int(frame_count_value)
frame_number = 0

# ==================== 板块 4：逐帧处理并保存 ====================
try:
    while success:
        frame = frame[0:video_height, 0:video_width].copy()

        # 正常情况下均分五段；拿不到总帧数时每两秒切换效果。
        if total_frames > 0:
            effect = min(4, int(frame_number * 5 / total_frames))
        else:
            effect = int(frame_number / fps / 2) % 5

        gray_face = cv2.cvtColor(small_face, cv2.COLOR_BGR2GRAY)
        if effect == 0:
            processed_face = small_face.copy()
            label = "Original"
        elif effect == 1:
            # 灰度图转回三通道，才能放到三通道视频画面里。
            processed_face = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)
            label = "Grayscale"
        elif effect == 2:
            processed_face = cv2.GaussianBlur(small_face, (15, 15), 0)
            label = "Blur"
        elif effect == 3:
            face_edges = cv2.Canny(gray_face, 100, 200)
            processed_face = cv2.cvtColor(face_edges, cv2.COLOR_GRAY2BGR)
            label = "Edges"
        else:
            processed_face = cv2.applyColorMap(gray_face, cv2.COLORMAP_JET)
            label = "Color Map"

        # 将处理后的脸部覆盖到视频左上角。
        frame[top:top + new_height, left:left + new_width] = processed_face
        cv2.putText(
            frame, label, (10, max(15, video_height - 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        writer.write(frame)
        frame_number = frame_number + 1
        success, frame = cap.read()
finally:
    # 出现错误时也要释放文件资源。
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

# ==================== 板块 5：检查输出 ====================
check = cv2.VideoCapture(output_video_path)
success, test_frame = check.read()
check.release()
if not success:
    raise RuntimeError("输出视频无法正常读取")
print("完成，写入帧数：", frame_number)
print("裁脸图片：", face_path)
print("输出视频：", output_video_path)
print("提示：OpenCV 写出的视频不包含原视频音轨。")
