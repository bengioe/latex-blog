from TexSoup import TexSoup, data
import bibtexparser as bib
import subprocess
import sys
import json
import os

print("Parsing", sys.argv[1])

soup = TexSoup(open(sys.argv[1], 'r').read())
print("Generating...")
bibdata = [None]

def parseaccents(s):
    return (s.replace('{\\\'o}', 'ó')
            .replace('{\\\'e}', 'é')
            .replace('{\\\'a}', 'á')
            .replace('{\\\'i}', 'í')
            .replace('{\\\'\\i}', 'í')
            .replace('{\\"e}', 'ë')
            .replace('{\\"a}', 'ä')
            .replace('{\\"o}', 'ö')
            .replace('{\\"u}', 'ü')
            .replace('{\\"i}', 'ï')
            .replace('{\\"\\i}', 'ï')
            .replace('{\\l}', 'ł')
            .replace('{\\k{e}}', 'ę')
            .replace('{\\c{c}}', 'ç')
            .replace('{\\aa}', 'å')
            )

def parsebibauthors(s):
    return [parseaccents(i).split(', ')[::-1] for i in s.split(' and ')]


def checkbib():
    entries = set()
    for i in bibdata[0].entries:
        eid = i['ID']
        if eid in entries:
            print('duplicate entry for', eid)
        entries.add(eid)

def bib2html(cid, inline=False):
    cids = [i.strip() for i in cid.split(',')]
    entries = []
    for cid in cids:
        for i in bibdata[0].entries:
            if i['ID'] in cid:
                entries.append(i)
                break
        else:
            print(ValueError(f"couldn't find {cid}"))
            return "(?)"
    links = []
    for ent in entries:
        auts = parsebibauthors(ent['author'])
        for i in auts:
            i = ' '.join(i)
            if '{' in i:
                print('Unparsed accent?', i)
        aut0 = ' '.join(auts[0])
        etal = ' et al.' if len(auts) > 1 else ''
        if inline:
            citestr = f"{aut0}{etal} ({ent['year']})"
        else:
            citestr = f"{aut0}{etal}, {ent['year']}"
        full = ", ".join(" ".join(i) for i in auts) + f'<br/><i>{ent["title"]}</i>, {ent["year"]}'
        url = (ent['url'] if 'url' in ent else
               f'https://scholar.google.com/scholar?q={"+".join(ent["title"].split())}&btnG=')
        href = f'<a class="tooltip" href="{url}"><span>{full}</span>{citestr}</a>'
        if 'posterurl' in ent:
            href += f'<sup><a href="{ent["posterurl"]}">[poster]</a></sup>'
        links.append(href)
    #sidestr = f'<div class="left-side">{full}</div>'
    html = (', ' if inline else '; ').join(links)
    if not inline:
        html = f'({html})'
    return html

# A cheap (bad) trick to have global variables is to wrap them in
# lists and always modify element 0.
num_canvas = [0]
title = [""]
body = []
script_hrefs = []

tex_to_render = []
class TexPromise:
    def __init__(self, idx=None, parts=None, join='', callback=None):
        self.idx = idx
        self.parts = parts
        self.join = join
        self.callback = callback

    def __call__(self):
        if self.idx is not None:
            return rendered_tex[self.idx]
        elif self.callback is not None:
            return self.callback()
        return self.join.join([i() if isinstance(i, TexPromise) else i for i in self.parts])

sections = []
flags = {
    'counters': {
        'section': 0,
        'subsection': 1,
    },
    'numbersections': False,
}

def make_toc():
    secs = []
    for i, s in enumerate(sections):
        if s[0] == 'section':
            secs.append(f'<li>{s[2]} <a href="#s{i+1}">{s[1]}</a></li>')
        if s[0] == 'subsection':
            secs.append(f'<li>{s[2]}.{s[3]} <a href="#s{i+1}">{s[1]}</a></li>')
    return f'<ul class="toc">{os.linesep.join(secs)}</ul>'

def proc(expr):
    #print(expr)
    #print(type(expr))
    def proc_sub(x, j=''):
        subs = [proc(i) for i in x.contents]
        if any(isinstance(i, TexPromise) for i in subs):
            return TexPromise(parts=subs, join=j)
        return j.join(subs)

    if isinstance(expr, data.TexEnv):
        display = True
        if expr.begin == '$':
            subexpr = str(expr)[1:-1]
            display = False
        elif expr.begin == '$$':
            subexpr = r'\begin{aligned}' + expr.string + r'\end{aligned}'
        elif expr.name == 'aligned':
            subexpr = str(expr)
        elif expr.name == 'itemize':
            return TexPromise(parts=('<ul>',proc_sub(expr),'</ul>'))
        else:
            raise ValueError(expr)
            subexpr = str(expr)
        tex_to_render.append((subexpr, display))
        return TexPromise(len(tex_to_render) - 1)
    elif isinstance(expr, data.TexText):
        return str(expr).replace('\n\n','<br/>\n')
    elif isinstance(expr, data.TexCmd):
        if expr.name == 'canvas0':
            return ('<canvas id="can0" width="800px" height="600px"></canvas>' +
                    '<script>cload("can0")</script>')
        elif expr.name == 'canvas':
            num_canvas[0] += 1
            s = (f'<div class="scontainer" style="width:{expr.args[1].string}px">'
                 f'<div id="can{num_canvas[0]}_div">'
                 f'<canvas id="can{num_canvas[0]}" width="{expr.args[1].string}px"'
                 f' height="{expr.args[2].string}px"></canvas></div></div>'
                 f'<script>{expr.args[0].string}("can{num_canvas[0]}")</script>')
            if len(expr.args) == 4:
                xtra = expr.args[3].string.split(',')
                if 'center' in xtra:
                    s = "<center>"+s+"</center>"
            return s
        elif expr.name == 'title':
            title[0] = expr.args[0].string
        elif expr.name == 'href':
            return f'<a href={expr.args[1].string}>{expr.args[0].string}</a>'
        elif expr.name == 'url':
            return f'<a href={expr.args[0].string}>{expr.args[0].string}</a>'
        elif expr.name == 'cite' or expr.name == 'citep':
            return bib2html(expr.args[0].string)
        elif expr.name == 'citet':
            return bib2html(expr.args[0].string, inline=True)
        elif expr.name == 'bibliography':
            print(f"Loading bibdata from {expr.args[0].string}")
            bibdata[0] = bib.load(open(expr.args[0].string, 'r'))
            checkbib()
            print(f"Done")
        elif expr.name == 'x':
            return ''
        elif expr.name == 'verbatim':
            return expr.args[0].string
        elif expr.name == 'item':
            return TexPromise(parts=('<li>',proc_sub(expr), '</li>'))
        elif expr.name == 'section':
            t = proc_sub(expr.args[0])
            flags['counters']['section'] += 1
            flags['counters']['subsection'] = 1
            sc = ((str(flags['counters']['section']),)
                  if flags['numbersections'] else [])
            sections.append(('section', t, *sc))
            return TexPromise(parts=(f'<a name="s{len(sections)}"></a><h3>', '.'.join(sc), ' ',
                                     t, '</h3>'))
        elif expr.name == 'subsection':
            t = proc_sub(expr.args[0])
            sc = ((str(flags['counters']['section']), str(flags['counters']['subsection']))
                  if flags['numbersections'] else [])
            sections.append(('subsection', t, *sc))
            flags['counters']['subsection'] += 1
            return TexPromise(parts=(f'<a name="s{len(sections)}"></a><h4>', '.'.join(sc), ' ',
                                     t, '</h4>'))
        elif expr.name == 'textbf':
            return TexPromise(parts=('<b>',proc_sub(expr.args[0]), '</b>'))
        elif expr.name == 'emph' or expr.name == 'textit':
            return TexPromise(parts=('<i>',proc_sub(expr.args[0]), '</i>'))
        elif expr.name == 'centered':
            return TexPromise(parts=('<center>',proc_sub(expr.args[0]), '</center>'))
        elif expr.name == 'tableofcontents':
            return TexPromise(callback=make_toc)
        elif expr.name == 'donumbersections':
            flags['numbersections'] = True
        elif expr.name == '&':
            return f'&amp;'
        elif expr.name == 'includejs':
            script_hrefs.append(f'<script src="{expr.args[0].string}" type="text/javascript"></script>')
        else:
            raise ValueError(expr.name)
    elif isinstance(expr, str):
        return expr
    elif isinstance(expr, data.OArg):
        return expr.string
    else:
        raise ValueError(expr)
    return ""

for i in soup.all:
    #print("-", repr(i), " %%")
    expr = i.expr
    u = proc(expr)
    body.append(u)

print("Rendering math...")
print(__file__)
src_dir = os.path.split(__file__)[0]
rendered_tex = json.loads(
    subprocess.check_output(f"node {src_dir}/tex2html_many.js",
                            input=json.dumps(tex_to_render).encode('utf8'),
                            shell=True).decode('utf8'))
body = "".join([i() if isinstance(i, TexPromise) else i
                for i in body])
print("Done")


out = f"""
<!DOCTYPE html>
<meta charset="utf-8">
<html>
  <head>
    <title>{title[0]}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.css" integrity="sha384-AfEj0r4/OFrOo5t7NnNe46zW/tFgW6x/bCJG8FqQCEo3+Aro6EYUG4+cU+KJWu/X" crossorigin="anonymous">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.js" integrity="sha384-g7c+Jr9ZivxKLnZTDUhnkOnsh30B4H0rpLUpJ4jAIKs4fnJI+sEnkvrMWph2EDg4" crossorigin="anonymous"></script>
    <script src="https://fpcdn.s3.amazonaws.com/apps/polygon-tools/0.4.6/polygon-tools.min.js" type="text/javascript"></script>
    {os.linesep.join(script_hrefs)}
    <link rel="stylesheet" href="main.css">
    <meta name="viewport" content="width=device-width, initial-scale=1">
  </head>
  <body>
   <div class="content">
   <center><a href="http://folinoid.com/">[Home]</a></center>
     {body}
   </div>
   <div style='height: 10em;'></div>
  </body>
</html>
"""
print("Writing output to", sys.argv[1]+".html")
with open(sys.argv[1]+".html", 'w') as f:
    f.write(out)
