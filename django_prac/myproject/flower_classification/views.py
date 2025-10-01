from django.shortcuts import render
from django.http import HttpResponse
from tensorflow.keras.models import load_model
import cv2
import numpy as np

def classify_flower(request):
    if request.method == 'POST' and request.FILES['image']:
        image_file = request.FILES['image']
        image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (128, 128))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        model = load_model('C:\\Users\\JayMa\\Downloads\\flower_images\\flower_images\\fnm.h5')
        prediction = model.predict(image)
        flower_type = np.argmax(prediction)
        
        # Map flower_type to flower_name using the dictionary
        flower_names = {
            0: "phlox",
            1: "rose",
            2: "calendula",
            3: "iris",
            4: "leucanthemum maximum (Shasta daisy)",
            5: "campanula (bellflower)",
            6: "viola",
            7: "rudbeckia laciniata (Goldquelle)",
            8: "peony",
            9: "aquilegia",
        }

        # Additional information for each flower
        flower_info = {
            "rose": {
                "best_season": "Spring",
                "temperature": "15-25°C",
                "water_requirements": "Moderate",
                "nutrients_needed": "Nitrogen, Phosphorus, Potassium",
                "additional_care": "Ensure the soil is well-drained and avoid overwatering. If neglected, the rose plant may become dry and susceptible to pests and diseases.",
            },
            # Add information for other flowers
        }
        
        flower_name = flower_names.get(flower_type, "Unknown")
        info = flower_info.get(flower_name.lower())
        if info:
            info_str = f"Best Season to Grow: {info['best_season']}<br>"
            info_str += f"Temperature: {info['temperature']}<br>"
            info_str += f"Water Requirements: {info['water_requirements']}<br>"
            info_str += f"Nutrients Needed: {info['nutrients_needed']}<br>"
            info_str += f"Additional Care: {info['additional_care']}"
        else:
            info_str = "Additional information not available for this flower."
        
        return HttpResponse(f"The predicted flower is: {flower_name}<br>{info_str}")
    else:
        return render(request, 'classify_flower.html')
