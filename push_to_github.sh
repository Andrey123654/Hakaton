#!/usr/bin/env bash
# push_to_github.sh — выгрузка репозитория на GitHub.
#
# Использование:
#   TOKEN=ghp_xxxxxxxx ./push_to_github.sh Andrey123654 phishing-hackathon
#   # или для уже созданного репозитория:
#   TOKEN=ghp_xxxxxxxx ./push_to_github.sh <владелец>/<репозиторий>
#
# Токен создаётся в GitHub: Settings → Developer settings → Personal access
# tokens → Tokens (classic) → Generate new token, достаточно scope "repo".
set -euo pipefail
cd "$(dirname "$0")"

TOKEN="${TOKEN:?задайте переменную TOKEN с personal access token}"
TARGET="${1:?укажите репозиторий: имя или owner/name}"
NAME="${2:-}"

if [ -n "$NAME" ]; then OWNER="$TARGET"; REPO="$NAME"; else OWNER=""; REPO="$TARGET"; fi

if [ -n "$OWNER" ]; then
  echo "== проверяю существование $OWNER/$REPO =="
  CODE=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: token $TOKEN" \
         "https://api.github.com/repos/$OWNER/$REPO")
  if [ "$CODE" = "404" ]; then
    echo "== создаю репозиторий $OWNER/$REPO =="
    curl -s -X POST -H "Authorization: token $TOKEN" \
         -H "Accept: application/vnd.github+json" \
         https://api.github.com/user/repos \
         -d "{\"name\":\"$REPO\",\"private\":false,\"description\":\"Phishing URL detection: public F1 = 0.996062 (Cyberhackai hackathon)\"}" \
         | head -c 300; echo
  fi
  URL="https://$TOKEN@github.com/$OWNER/$REPO.git"
else
  # имя пользователя определим по токену
  OWNER=$(curl -s -H "Authorization: token $TOKEN" https://api.github.com/user \
          | sed -n 's/.*"login": *"\([^"]*\)".*/\1/p' | head -1)
  echo "== пользователь: $OWNER, создаю репозиторий $REPO =="
  curl -s -X POST -H "Authorization: token $TOKEN" \
       -H "Accept: application/vnd.github+json" \
       https://api.github.com/user/repos \
       -d "{\"name\":\"$REPO\",\"private\":false,\"description\":\"Phishing URL detection: public F1 = 0.996062 (Cyberhackai hackathon)\"}" \
       | head -c 300; echo
  URL="https://$TOKEN@github.com/$OWNER/$REPO.git"
fi

git remote remove origin 2>/dev/null || true
git remote add origin "$URL"
git branch -M main
echo "== отправляю $(git ls-files | wc -l | tr -d ' ') файлов (~64 МБ) =="
git push -u origin main
echo
echo "== готово: https://github.com/$OWNER/$REPO =="
