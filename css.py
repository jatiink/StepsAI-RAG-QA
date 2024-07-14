import base64

#CSS
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url(data:image/png;base64,{base64.b64encode(open("pexels-codioful-6985001.jpg", "rb").read()).decode()});
    background-size: 220%;
    color: white;
    background-position: top left;
    background-repeat: repeat;
    background-attachment: local;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    color: white;
}}

h1, h2 {{
    color: white;
}}

[data-testid=stNotificationContentError] {{
    color: white;
}}

.st-ay {{
    background-color: rgb(96, 0, 160);
    border-radius: 10px;
    border-color: rgb(96, 0, 160);
}}

div.stButton > button:first-child {{
    background-color: rgb(96, 0, 160);
    color: white;
    font-size: 20px;
    height: 2em;
    width: 8em;
    border-radius: 10px;
    transition: background-color 0.3s, color 0.3s;
}}

div.stButton > button:first-child:hover,
div.stButton > button:first-child:active,
div.stButton > button:first-child:focus {{
    background-color: rgb(96, 0, 160) !important;
    color: white !important;
    border-color: white ;
}}

.st-emotion-cache-1mnnx3i:focus:not(:active) {{
    border-color: white;
    color: white;
    height: 1.1em;
    width: 10.1em;
}}


[data-testid=baseButton-secondary] {{
    background-color: rgb(255, 255, 255);
    color: rgb(96, 0, 160);
    font-size: 20px;
    height: 2em;
    width: 10em;
    border-radius: 10px;
}}

.stTextInput > div > div > input {{
    background-color: rgb(96, 0, 160);
    border-radius: 10px;
}}

.answer-card {{
    background-color: white;
    color: black;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}}

.answer-card h3 {{
    color: black;
}}
</style>
"""