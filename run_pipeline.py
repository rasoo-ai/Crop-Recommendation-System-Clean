import subprocess

print("=" * 60)
print("🚀 ONE CLICK GIS PIPELINE STARTED")
print("=" * 60)

scripts = [
    "scripts/09_normalize_names.py",
    "scripts/10_exact_matching.py",
    "scripts/12_final_dataset.py"
]

for script in scripts:
    print("\n" + "="*60)
    print(f"RUNNING: {script}")
    print("="*60)

    result = subprocess.run(["python", script])

    if result.returncode != 0:
        print(f"❌ ERROR in {script}")
        print("Stopping pipeline...")
        exit(1)

print("\n" + "=" * 60)
print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)
