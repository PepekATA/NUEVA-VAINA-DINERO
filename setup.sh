mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = \$PORT\n\
[theme]\n\
base=\"dark\"\n\
primaryColor=\"#4CAF50\"\n\
backgroundColor=\"#1a1a1a\"\n\
secondaryBackgroundColor=\"#2a2a2a\"\n\
textColor=\"#ffffff\"\n\
" > ~/.streamlit/config.toml
