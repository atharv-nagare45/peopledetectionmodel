from ultralytics import YOLO

if __name__ == "__main__":

    # Load trained model
    model = YOLO(
        r"runs\detect\water_rescue_person\person_model_gpu-2\weights\best.pt"
    )

    # Validate model
    results = model.val(
        data=r"data.yaml",
        split="val",
        plots=True,
        workers=0
    )

    print("\n========== VALIDATION RESULTS ==========")
    print("mAP50     :", results.box.map50)
    print("mAP50-95  :", results.box.map)
    print("Precision :", results.box.mp)
    print("Recall    :", results.box.mr)
    print("========================================")
    