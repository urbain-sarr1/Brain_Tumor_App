import subprocess
import sys

REPO_URL = "https://github.com/urbain-sarr1/Brain_Tumor_App.git"
COMMIT_MESSAGE = "Final Version"


def run(command, capture=False):
    """Exécute une commande. Arrête le script en cas d'erreur."""
    print(f"\n>>> {command}")
    result = subprocess.run(command, shell=True, capture_output=capture, text=True)

    if result.returncode != 0:
        print(f"\n❌ Erreur avec : {command}")
        if capture and result.stderr:
            print(result.stderr)
        sys.exit(result.returncode)

    return result


print("🚀 Envoi du projet Brain-Tumor-App vers GitHub")

# 1. Initialisation Git
run("git init")

# 2. Dépôt distant
remote = subprocess.run("git remote get-url origin", shell=True, capture_output=True, text=True)

if remote.returncode == 0:
    print("🔄 Dépôt GitHub déjà configuré.")
    run(f"git remote set-url origin {REPO_URL}")
else:
    print("🔗 Ajout du dépôt GitHub.")
    run(f"git remote add origin {REPO_URL}")

# 3. Ajouter les fichiers
run("git add .")

# 4. Commit (uniquement s'il y a des changements)
status = run("git status --porcelain", capture=True)

if status.stdout.strip():
    run(f'git commit -m "{COMMIT_MESSAGE}"')
else:
    print("\nℹ️ Rien à commiter, on passe au push.")

# 5. Branche principale
run("git branch -M main")

# 6. Push
run("git push -u origin main")

print("\n" + "=" * 60)
print("✅ PROJET ENVOYÉ SUR GITHUB !")
print("=" * 60)
print("🌐 https://github.com/urbain-sarr1/Brain_Tumor_App")