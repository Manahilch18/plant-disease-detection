"""
disease_info.py

Disease information for all 38 PlantVillage classes.

Project:
    Plant Disease Detection

Note:
    This information is intended for educational/informational purposes.
    It should not be treated as a definitive agricultural diagnosis.
"""

from __future__ import annotations


DISEASE_INFO = {
    # ============================================================
    # APPLE
    # ============================================================

    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "description": (
            "A fungal disease that causes olive-green to brown lesions "
            "on apple leaves and fruit."
        ),
        "cause": "Fungus: Venturia inaequalis.",
        "management": (
            "Remove infected plant material, improve air circulation, "
            "and use appropriate fungicide programs when necessary."
        ),
    },

    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot",
        "description": (
            "A fungal disease that produces dark leaf spots and can "
            "cause fruit rot."
        ),
        "cause": "Fungus: Botryosphaeria obtusa.",
        "management": (
            "Remove mummified fruit and infected branches and maintain "
            "good orchard sanitation."
        ),
    },

    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "description": (
            "A fungal disease that causes yellow-orange spots on "
            "apple leaves and may affect fruit."
        ),
        "cause": "Fungus: Gymnosporangium species.",
        "management": (
            "Remove nearby alternate hosts where practical and use "
            "appropriate disease-management practices."
        ),
    },

    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "description": (
            "The image shows characteristics associated with a healthy "
            "apple leaf."
        ),
        "cause": "No disease detected.",
        "management": (
            "Continue good plant nutrition, watering, sanitation, "
            "and regular monitoring."
        ),
    },

    # ============================================================
    # BLUEBERRY
    # ============================================================

    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "description": (
            "The image shows characteristics associated with a healthy "
            "blueberry leaf."
        ),
        "cause": "No disease detected.",
        "management": (
            "Maintain appropriate irrigation, nutrition, sanitation, "
            "and regular plant monitoring."
        ),
    },

    # ============================================================
    # CHERRY
    # ============================================================

    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "description": (
            "A fungal disease that produces white powdery growth on "
            "leaves and young plant tissues."
        ),
        "cause": "Powdery mildew fungi.",
        "management": (
            "Improve air circulation, avoid excessive humidity around "
            "foliage, and use suitable fungicide management when needed."
        ),
    },

    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "description": (
            "The image shows characteristics associated with a healthy "
            "cherry leaf."
        ),
        "cause": "No disease detected.",
        "management": (
            "Maintain proper watering, nutrition, sanitation, and "
            "regular monitoring."
        ),
    },

    # ============================================================
    # CORN
    # ============================================================

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn",
        "disease": "Cercospora Leaf Spot / Gray Leaf Spot",
        "description": (
            "A fungal disease that produces elongated gray or brown "
            "lesions on corn leaves."
        ),
        "cause": "Fungus: Cercospora species.",
        "management": (
            "Use resistant varieties where available, manage crop "
            "residue, rotate crops, and use fungicides when appropriate."
        ),
    },

    "Corn_(maize)___Common_rust_": {
        "plant": "Corn",
        "disease": "Common Rust",
        "description": (
            "A fungal disease characterized by reddish-brown rust "
            "colored pustules on corn leaves."
        ),
        "cause": "Fungus: Puccinia sorghi.",
        "management": (
            "Use resistant varieties and monitor fields; fungicide "
            "treatment may be considered when disease pressure is high."
        ),
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn",
        "disease": "Northern Leaf Blight",
        "description": (
            "A fungal disease producing large elongated gray-green "
            "lesions on corn leaves."
        ),
        "cause": "Fungus: Exserohilum turcicum.",
        "management": (
            "Use resistant hybrids, rotate crops, manage residue, "
            "and consider fungicides when appropriate."
        ),
    },

    "Corn_(maize)___healthy": {
        "plant": "Corn",
        "disease": "Healthy",
        "description": (
            "The image shows characteristics associated with a healthy "
            "corn leaf."
        ),
        "cause": "No disease detected.",
        "management": (
            "Maintain balanced nutrition, proper irrigation, sanitation, "
            "and regular field monitoring."
        ),
    },

    # ============================================================
    # GRAPE
    # ============================================================

    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "description": (
            "A fungal disease that causes brown or reddish leaf lesions "
            "and dark fruit rot."
        ),
        "cause": "Fungus: Guignardia bidwellii.",
        "management": (
            "Remove infected plant material, improve canopy airflow, "
            "and apply suitable fungicide programs when necessary."
        ),
    },

    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca (Black Measles)",
        "description": (
            "A complex grapevine disease associated with wood-infecting "
            "fungi and characteristic leaf and fruit symptoms."
        ),
        "cause": "Several wood-colonizing fungal pathogens.",
        "management": (
            "Use healthy planting material, remove severely affected "
            "wood where practical, and maintain vineyard sanitation."
        ),
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf Blight (Isariopsis Leaf Spot)",
        "description": (
            "A fungal leaf disease that produces dark spots and may "
            "cause affected leaves to deteriorate."
        ),
        "cause": "Fungal pathogen associated with Isariopsis leaf spot.",
        "management": (
            "Remove infected leaves where practical, improve airflow, "
            "and use appropriate fungicide management."
        ),
    },

    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "description": (
            "The image shows characteristics associated with a healthy "
            "grape leaf."
        ),
        "cause": "No disease detected.",
        "management": (
            "Maintain good vineyard sanitation, irrigation, nutrition, "
            "and regular monitoring."
        ),
    },

    # ============================================================
    # PEACH
    # ============================================================

    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "description": (
            "A bacterial disease that causes small dark spots on leaves "
            "and may affect fruit."
        ),
        "cause": "Bacterium: Xanthomonas species.",
        "management": (
            "Use resistant varieties where available, avoid unnecessary "
            "leaf wetness, maintain sanitation, and follow appropriate "
            "bacterial disease-management practices."
        ),
    },

    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "description": (
            "The image shows characteristics associated with a healthy "
            "peach leaf."
        ),
        "cause": "No disease detected.",
        "management": (
            "Maintain appropriate irrigation, nutrition, sanitation, "
            "and regular monitoring."
        ),
    },

    # ============================================================
    # PEPPER
    # ============================================================

    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "disease": "Bacterial Spot",
        "description": (
            "A bacterial disease causing small dark lesions on pepper "
            "leaves and fruit."
        ),
        "cause": "Xanthomonas species.",
        "management": (
            "Use clean seed and planting material, avoid overhead "
            "irrigation, remove infected material, and follow suitable "
            "disease-management practices."
        ),
    },

    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper",
        "disease": "Healthy",
        "description": (
            "The image shows characteristics associated with a healthy "
            "bell pepper leaf."
        ),
        "cause": "No disease detected.",
        "management": (
            "Maintain proper watering, nutrition, sanitation, and "
            "regular plant monitoring."
        ),
    },

    # ============================================================
    # POTATO
    # ============================================================

    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "description": (
            "A fungal disease that commonly produces circular brown "
            "lesions with concentric rings on potato leaves."
        ),
        "cause": "Fungus: Alternaria solani.",
        "management": (
            "Use crop rotation, remove infected debris, maintain plant "
            "vigor, and use appropriate fungicides when necessary."
        ),
    },

    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "description": (
            "A destructive disease that causes dark water-soaked "
            "lesions on leaves and stems."
        ),
        "cause": "Oomycete: Phytophthora infestans.",
        "management": (
            "Use resistant varieties where available, remove infected "
            "material, monitor weather conditions, and use appropriate "
            "fungicide programs."
        ),
    },

    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "description": (
            "The image shows characteristics associated with a healthy "
            "potato leaf."
        ),
        "cause": "No disease detected.",
        "management": (
            "Maintain suitable irrigation, nutrition, sanitation, and "
            "regular crop monitoring."
        ),
    },

    # ============================================================
    # RASPBERRY
    # ============================================================

    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "description": (
            "The image shows characteristics associated with a healthy "
            "raspberry leaf."
        ),
        "cause": "No disease detected.",
        "management": (
            "Maintain proper watering, nutrition, pruning, sanitation, "
            "and regular monitoring."
        ),
    },

    # ============================================================
    # SOYBEAN
    # ============================================================

    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "description": (
            "The image shows characteristics associated with a healthy "
            "soybean leaf."
        ),
        "cause": "No disease detected.",
        "management": (
            "Maintain appropriate irrigation, nutrition, field hygiene, "
            "and regular crop monitoring."
        ),
    },

    # ============================================================
    # SQUASH
    # ============================================================

    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "description": (
            "A fungal disease characterized by white powdery growth "
            "on leaf surfaces."
        ),
        "cause": "Powdery mildew fungi.",
        "management": (
            "Improve airflow, avoid prolonged leaf wetness, remove "
            "severely affected material, and use suitable fungicides "
            "when necessary."
        ),
    },

    # ============================================================
    # STRAWBERRY
    # ============================================================

    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "description": (
            "A fungal disease that produces dark purple to brown spots "
            "and scorched areas on strawberry leaves."
        ),
        "cause": "Fungus: Diplocarpon species.",
        "management": (
            "Remove infected leaves, improve airflow, avoid excessive "
            "leaf wetness, and use suitable disease-management practices."
        ),
    },

    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "description": (
            "The image shows characteristics associated with a healthy "
            "strawberry leaf."
        ),
        "cause": "No disease detected.",
        "management": (
            "Maintain good irrigation, nutrition, sanitation, and "
            "regular monitoring."
        ),
    },

    # ============================================================
    # TOMATO
    # ============================================================

    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "description": (
            "A bacterial disease producing small dark lesions on "
            "tomato leaves, stems, and fruit."
        ),
        "cause": "Xanthomonas species.",
        "management": (
            "Use clean seed, avoid overhead irrigation, remove infected "
            "material, and follow appropriate disease-management practices."
        ),
    },

    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "description": (
            "A fungal disease that produces dark lesions with "
            "concentric rings on tomato leaves."
        ),
        "cause": "Fungus: Alternaria solani.",
        "management": (
            "Use crop rotation, remove infected debris, maintain good "
            "airflow, and apply suitable fungicides when necessary."
        ),
    },

    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "description": (
            "A serious disease that produces dark water-soaked lesions "
            "on tomato foliage and fruit."
        ),
        "cause": "Oomycete: Phytophthora infestans.",
        "management": (
            "Use resistant varieties where available, remove infected "
            "material, monitor disease conditions, and use appropriate "
            "fungicide programs."
        ),
    },

    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "description": (
            "A fungal disease that commonly causes pale or yellow areas "
            "on the upper leaf surface and mold growth underneath."
        ),
        "cause": "Fungus: Passalora fulva.",
        "management": (
            "Improve greenhouse ventilation, reduce humidity, remove "
            "infected leaves, and use suitable fungicide management."
        ),
    },

    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "description": (
            "A fungal disease characterized by numerous small circular "
            "spots with dark borders on tomato leaves."
        ),
        "cause": "Fungus: Septoria lycopersici.",
        "management": (
            "Remove infected debris, avoid overhead irrigation, improve "
            "airflow, and use suitable fungicide programs when necessary."
        ),
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Two-Spotted Spider Mite",
        "description": (
            "A pest infestation that can cause stippling, yellowing, "
            "bronzing, and webbing on tomato leaves."
        ),
        "cause": "Two-spotted spider mite: Tetranychus urticae.",
        "management": (
            "Monitor plants regularly, conserve beneficial predators, "
            "and use appropriate mite-management practices when needed."
        ),
    },

    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "description": (
            "A fungal disease producing brown circular lesions that "
            "may develop concentric rings."
        ),
        "cause": "Fungus: Corynespora cassiicola.",
        "management": (
            "Improve airflow, reduce leaf wetness, remove infected "
            "material, and use appropriate fungicide management."
        ),
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "description": (
            "A viral disease that can cause yellowing, upward curling "
            "of leaves, and reduced plant growth."
        ),
        "cause": (
            "Tomato yellow leaf curl virus, commonly spread by "
            "whiteflies."
        ),
        "management": (
            "Control whitefly vectors, remove infected plants where "
            "appropriate, use resistant varieties, and manage weeds "
            "that may harbor vectors."
        ),
    },

    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "description": (
            "A viral disease that can produce mosaic patterns, leaf "
            "distortion, and reduced plant growth."
        ),
        "cause": "Tomato mosaic virus.",
        "management": (
            "Use clean seed and tools, remove infected plants, control "
            "mechanical spread, and maintain good sanitation."
        ),
    },

    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "description": (
            "The image shows characteristics associated with a healthy "
            "tomato leaf."
        ),
        "cause": "No disease detected.",
        "management": (
            "Maintain proper irrigation, nutrition, sanitation, and "
            "regular monitoring."
        ),
    },
}


def get_disease_info(class_name: str) -> dict:
    """
    Return disease information for a PlantVillage class.

    Parameters
    ----------
    class_name:
        Exact PlantVillage class name.

    Returns
    -------
    dict
        Disease information.
    """

    return DISEASE_INFO.get(
        class_name,
        {
            "plant": "Unknown",
            "disease": class_name.replace("_", " "),
            "description": "Information is not available for this disease.",
            "cause": "Unknown.",
            "management": (
                "Consult a plant disease specialist for further "
                "diagnosis and management."
            ),
        },
    )