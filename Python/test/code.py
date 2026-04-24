import qrcode

img = qrcode.make("ST1001")
img.save("student.png")