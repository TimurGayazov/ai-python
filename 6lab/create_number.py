from PIL import Image, ImageDraw, ImageFont


def create_digit_image(digit, image_path):
    # Создаем пустое изображение размером 28x28 пикселей
    img = Image.new('L', (28, 28), color=0)  # 'L' для серого изображения, цвет 0 - черный

    # Используем ImageDraw для рисования на изображении
    draw = ImageDraw.Draw(img)

    # Настраиваем шрифт (можно использовать системный шрифт или любой другой)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # Получаем размеры текста
    text_bbox = draw.textbbox((0, 0), str(digit), font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Вычисляем координаты для центрирования текста
    text_x = (28 - text_width) // 2
    text_y = (28 - text_height) // 2

    # Рисуем цифру на изображении
    draw.text((text_x, text_y), str(digit), font=font, fill=255)  # Цвет 255 - белый

    # Сохраняем изображение
    img.save(image_path)


for i in range(0, 10):
    create_digit_image(i, f'./img/digit_{i}.png')
