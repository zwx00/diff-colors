import json

def get_css_colors():
    css_colors = {}
    with open("data/css-colors.json", "r") as f:
        for name, rgb in json.load(f).items():
            css_colors[name] = tuple(int(c) / 255.0 for c in rgb.split())
    return css_colors

def get_xkcd_colors():
    xkcd_colors = {}
    with open("data/xkcd-colors.json", "r") as f:
        for name, rgb in json.load(f).items():
            r, g, b = rgb.split("#")[1][:2], rgb.split("#")[1][2:4], rgb.split("#")[1][4:]
            xkcd_colors[name] = tuple(int(c, 16) / 255.0 for c in [r, g, b])
    return xkcd_colors
