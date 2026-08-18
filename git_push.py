import subprocess
import sys


REPO_URL = "https://github.com/urbain-sarr1/Brain_Tumor_App.git"


def run(command):
    print(f"\n>>> {command}")

    result = subprocess.run(
        command,
        shell=True
    )

    if result.returncode != 0:
        print(f"\n❌ Erreur avec : {command}")
        sys.exit(result.returncode)


print("🚀 Envoi du projet Brain-Tumor-App vers GitHub")


# ============================================================
# 1. Initialisation Git
# ============================================================

run("git init")


# ============================================================
# 2. Configuration du dépôt distant
# ============================================================

result = subprocess.run(
    "git remote get-url origin",
    shell=True,
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("🔄 Dépôt GitHub déjà configuré.")
    run(f"git remote set-url origin {REPO_URL}")
else:
    print("🔗 Ajout du dépôt GitHub.")
    run(f"git remote add origin {REPO_URL}")


# ============================================================
# 3. Ajouter les fichiers
# ============================================================

run("git add .")


# ============================================================
# 4. Vérifier
# ============================================================

run("git status")


# ============================================================
# 5. Commit
# ============================================================

run('git commit -m "Correction de bug memoire"')


# ============================================================
# 6. Branche principale
# ============================================================

run("git branch -M main")


# ============================================================
# 7. Envoyer vers GitHub
# ============================================================

run("git push -u origin main")


print("\n" + "=" * 60)
print("✅ PROJET ENVOYÉ SUR GITHUB !")
print("=" * 60)
print("🌐 https://github.com/urbain-sarr1/Brain_Tumor_App")