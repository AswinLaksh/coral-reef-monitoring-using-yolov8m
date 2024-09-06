from ultralytics import YOLO # type: ignore
import time

model=YOLO('best.pt')

img_arr=['dataset/1347803447_4834a50509_o.jpg',
         "dataset/1558786009_097a857b64_b.jpg",
         "dataset/2980775197_e4526a5fe0_o.jpg",
         "dataset/1063594275_418daec726_b.jpg",
         "dataset/2901229228_8354d571d9_o.jpg",
         "dataset/12337435893_f2f44ea3f1_o.jpg",
         "dataset/12440017704_068016572a_o.jpg",
         "dataset/5815350035_8f2f1f2511_o.jpg",
         "dataset/4819583169_106ff643c9_o.jpg",
         "dataset/5815918114_9a94837b24_o.jpg",
         "dataset/4820197456_762199166b_o.jpg",
         "dataset/4892454919_a0e8181a43_b.jpg"
         ]

for img in img_arr:
    model.predict(source=img,show=True,save=False,conf=0.5)
    time.sleep(5)