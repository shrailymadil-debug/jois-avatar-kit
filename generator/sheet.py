import subprocess, json, sys
from PIL import Image, ImageDraw, ImageFont
EMO = {
 "joy": "mouthSmileLeft=0.9,mouthSmileRight=0.9,cheekSquintLeft=0.6,cheekSquintRight=0.6,eyeSquintLeft=0.45,eyeSquintRight=0.45,browInnerUp=0.1",
 "surprise": "browInnerUp=0.9,browOuterUpLeft=0.9,browOuterUpRight=0.9,eyeWideLeft=0.9,eyeWideRight=0.9,jawOpen=0.3,mouthFunnel=0.3",
 "focus": "browDownLeft=0.7,browDownRight=0.7,eyeSquintLeft=0.45,eyeSquintRight=0.45,mouthPressLeft=0.5,mouthPressRight=0.5",
 "empathy": "browInnerUp=0.9,mouthFrownLeft=0.4,mouthFrownRight=0.4,eyeSquintLeft=0.25,eyeSquintRight=0.25,mouthPressLeft=0.2,mouthPressRight=0.2",
}
VIS = ["sil","PP","FF","TH","DD","kk","CH","SS","nn","RR","aa","E","ih","oh","ou"]
tiles = [("viseme_"+v, f"viseme_{v}=1") for v in VIS] + [(k, s) for k, s in EMO.items()] + [("eyeBlink", "eyeBlinkLeft=1,eyeBlinkRight=1")]
T = 300; cols = 5; rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (cols * T, rows * (T + 34)), (24, 30, 29))
d = ImageDraw.Draw(sheet)
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18)
except Exception: font = ImageFont.load_default()
for i, (name, spec) in enumerate(tiles):
    out = f"/tmp/claude-0/t_{i}.png"
    subprocess.run(["python3", "preview.py", out, spec, "0.085"], check=True)
    im = Image.open(out).resize((T, T))
    x, y = (i % cols) * T, (i // cols) * (T + 34)
    sheet.paste(im, (x, y)); d.text((x + 10, y + T + 7), name, fill=(220, 236, 232), font=font)
sheet.save(sys.argv[1])
