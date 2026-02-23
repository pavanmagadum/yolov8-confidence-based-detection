from ultralytics import YOLO

# Load the model
model = YOLO('best.pt')

# Display model information
print("=" * 50)
print("MODEL INFORMATION")
print("=" * 50)

# Get model details
print(f"\nModel Type: {model.task}")
print(f"Model Name: {model.ckpt_path}")

# Get class names
print(f"\nNumber of Classes: {len(model.names)}")
print("\nClass Names:")
for idx, name in model.names.items():
    print(f"  {idx}: {name}")

# Display model summary
print("\n" + "=" * 50)
print("MODEL ARCHITECTURE SUMMARY")
print("=" * 50)
model.info()