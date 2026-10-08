"""Плеер записей для веб-интерфейса: видео с разметкой + таймлайн циклов/событий + живая панель выходов классификаторов.

Самодостаточный HTML-компонент (видео вшито как data-URI, внешних библиотек нет), поэтому работает без дополнительных флагов Streamlit:
  * таймлайн по людям: циклы цветом по оценке (синий — низкая, красный — высокая), события — красной рамкой; клик — перемотка, двойной клик по циклу — зацикливание;
  * панель справа показывает выходы всех моделей для цикла под курсором (итог, дешёвый пакет, фото-модель, VLM, метка), список событий;
  * клавиши: Пробел — пауза, ←/→ — кадр, Shift+←/→ — 1 с, , и . — предыдущий/следующий цикл, L — зацикливание цикла, [ и ] — скорость.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

MAX_EMBED_MB = 70.0

_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><style>
:root{--bg:#0f1115;--panel:#171a21;--ink:#e8e8e6;--mut:#9a9a96;--grid:#2a2e38;--acc:#4da3ff;--hot:#ff5a4d}
@media (prefers-color-scheme: light){:root{--bg:#f6f6f4;--panel:#ffffff;--ink:#1a1a19;--mut:#6b6b66;--grid:#dcdcd6;--acc:#1f6fd1;--hot:#d33a2e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.35 system-ui,Segoe UI,Roboto,sans-serif;outline:none}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:10px;padding:8px}
@media (max-width:760px){.wrap{grid-template-columns:1fr}}
.vw{position:relative;background:#000;border-radius:6px;overflow:hidden}
video{width:100%;max-height:460px;background:#000;display:block}
#ov{position:absolute;left:0;top:0;pointer-events:none}
.bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:6px 0}
button{background:var(--panel);color:var(--ink);border:1px solid var(--grid);border-radius:6px;padding:4px 9px;cursor:pointer;font:inherit}
button:hover{border-color:var(--acc)}button.on{border-color:var(--acc);color:var(--acc)}
canvas{width:100%;display:block;background:var(--panel);border:1px solid var(--grid);border-radius:6px;cursor:pointer}
.side{background:var(--panel);border:1px solid var(--grid);border-radius:8px;padding:10px;min-height:120px;max-height:78vh;overflow:auto}
.side h4{margin:0 0 6px;font-size:13px}.row{display:flex;align-items:center;gap:6px;margin:3px 0}.row .n{width:96px;color:var(--mut)}
.meter{flex:1;height:9px;background:var(--grid);border-radius:5px;overflow:hidden}.meter i{display:block;height:100%;background:var(--acc)}
.val{width:40px;text-align:right;font-variant-numeric:tabular-nums}.tag{display:inline-block;padding:1px 7px;border-radius:9px;border:1px solid var(--grid);color:var(--mut);margin-left:4px}
.er{margin:3px 0;padding:4px 6px;border-radius:6px;border:1px solid var(--grid);cursor:pointer}.er:hover{border-color:var(--hot)}
.ev{margin:3px 0;padding:4px 6px;border-radius:6px;border:1px solid var(--grid);cursor:pointer}.ev:hover{border-color:var(--hot)}.mut{color:var(--mut)}
</style></head><body tabindex="0"><div class="wrap"><div>
<div class="vw"><video id="v" controls preload="metadata" playsinline __SRC__></video><canvas id="ov"></canvas></div><div id="verr" class="mut"></div>
<div class="bar"><button id="pp">▶/⏸</button><button id="bk">⏮ кадр</button><button id="fw">кадр ⏭</button><button id="pc">◀ цикл</button><button id="nc">цикл ▶</button>
<button id="lp">⟲ цикл</button><button id="bx" class="on">рамки</button><button id="pe">◀ ошибка</button><button id="ne">ошибка ▶</button><span class="mut">скорость</span><button id="sl">−</button><b id="sp">1×</b><button id="sf">+</button><span class="mut" id="tm"></span></div>
<canvas id="tl" height="__H__"></canvas>
<div class="mut" style="margin-top:4px">клик — перемотка · двойной клик по циклу — зацикливание · цвет цикла = оценка · красная рамка = событие · зелёная = эталон · <b>E</b>/<b>Shift+E</b> — следующая/предыдущая ошибка</div>
</div><div class="side"><div id="cur"></div><hr style="border:0;border-top:1px solid var(--grid);margin:8px 0"><h4>События</h4><div id="evs"></div><div id="errh"></div><div id="errs"></div></div></div>
<script>
const D=__DATA__;const v=document.getElementById('v');v.addEventListener('error',()=>{const e=v.error,m=document.getElementById('verr');m.textContent='Видео не воспроизводится в этом браузере (код '+(e?e.code:'?')+(e&&e.message?': '+e.message:'')+'). ';const s=v.currentSrc||v.src;if(s&&!s.startsWith('data:')){const a=document.createElement('a');a.href=s;a.target='_blank';a.textContent='Открыть файл отдельно';m.appendChild(a)}});const _v=v,cv=document.getElementById('tl'),cx=cv.getContext('2d');
D.gt=D.gt||[];D.errors=D.errors||[];D.tracks=D.tracks||{};const ntop=(D.gt.length?1:0)+(D.errors.length?1:0);
const fps=D.fps||15,off=D.offset||0,tids=D.tids,rowH=22,pad=18;function ry(i){return pad+(ntop+i)*rowH}let showBox=true;let dur=D.duration||0;const now=()=>v.currentTime+off;let loop=null,speeds=[0.1,0.25,0.5,1,1.5,2,4],si=3;
function css(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim()}
function col(s){if(s==null||isNaN(s))return '#888';const a=[[60,120,200],[240,190,60],[230,70,60]],x=Math.max(0,Math.min(1,s))*2,i=Math.min(1,Math.floor(x)),f=x-i;
const c=a[i].map((p,k)=>Math.round(p+(a[i+1][k]-p)*f));return `rgb(${c})`}
function W(){const r=cv.getBoundingClientRect();return r.width}
function X(t){return 46+(W()-54)*Math.max(0,Math.min(1,(t-off)/(dur||1)))}
function T(x){return off+Math.max(0,Math.min(dur,(x-46)/(W()-54)*(dur||1)))}
function size(){const r=window.devicePixelRatio||1,w=cv.getBoundingClientRect().width,h=pad+rowH*(ntop+Math.max(1,tids.length))+6;cv.width=w*r;cv.height=h*r;cv.style.height=h+'px';cx.setTransform(r,0,0,r,0,0)}
function draw(){const w=W(),h=pad+rowH*(ntop+Math.max(1,tids.length))+6;cx.clearRect(0,0,w,h);cx.font='11px system-ui';cx.fillStyle=css('--mut');
for(let s=off;s<=off+dur;s+=Math.max(1,Math.round(dur/10))){cx.fillText(s.toFixed(0)+'с',X(s)-8,11);cx.fillStyle=css('--grid');cx.fillRect(X(s),pad-4,1,h);cx.fillStyle=css('--mut')}
let lane=0;if(D.gt.length){const y=pad;cx.fillStyle=css('--mut');cx.fillText('эталон',2,y+14);D.gt.forEach(g=>{cx.fillStyle=g.label==='POSITIVE'?'#2fb36b':g.label==='IGNORE'?'#e0a01b':'#8a96a3';cx.globalAlpha=.85;cx.fillRect(X(g.start),y+2,Math.max(3,X(g.end)-X(g.start)),rowH-6);cx.globalAlpha=1});lane=1}
if(D.errors.length){const y=pad+lane*rowH;cx.fillStyle=css('--mut');cx.fillText('ошибки',2,y+14);D.errors.forEach(e=>{cx.fillStyle=e.kind==='FP'?css('--hot'):'#e0a01b';const x0=X(e.start),x1=X(e.end==null?e.start:e.end);cx.fillRect(x0,y+5,Math.max(5,x1-x0),rowH-10);cx.fillStyle='#fff';cx.font='bold 9px system-ui';cx.fillText(e.kind,x0+1,y+rowH-7);cx.font='11px system-ui'})}
tids.forEach((id,i)=>{const y=ry(i);cx.fillStyle=css('--mut');cx.fillText('ID'+id,4,y+14)});
D.cycles.forEach((c,k)=>{const y=ry(tids.indexOf(c.tid));cx.fillStyle=col(c.score);cx.globalAlpha=.9;cx.fillRect(X(c.start),y+2,Math.max(2,X(c.end)-X(c.start)),rowH-6);cx.globalAlpha=1;
if(loop===k){cx.strokeStyle=css('--acc');cx.lineWidth=2;cx.strokeRect(X(c.start),y+1,Math.max(2,X(c.end)-X(c.start)),rowH-4)}});
D.events.forEach(e=>{const y=ry(Math.max(0,tids.indexOf(e.tid)));cx.strokeStyle=css('--hot');cx.lineWidth=2.5;cx.strokeRect(X(e.start)-1,y,Math.max(3,X(e.end)-X(e.start))+2,rowH-2)});
cx.fillStyle='#fff';cx.fillRect(X(now())-1,pad-4,2,h);cx.strokeStyle='#000';cx.lineWidth=.5;cx.strokeRect(X(now())-1,pad-4,2,h)}
function fmt(x){return x==null||isNaN(x)?'—':x.toFixed(2)}
function meter(n,x){return `<div class="row"><span class="n">${n}</span><span class="meter"><i style="width:${x==null||isNaN(x)?0:Math.round(x*100)}%"></i></span><span class="val">${fmt(x)}</span></div>`}
function panel(){const t=now();let a=D.cycles.map((c,k)=>[c,k]).filter(([c])=>t>=c.start&&t<=c.end);let note='';
if(!a.length){const p=D.cycles.map((c,k)=>[c,k]).filter(([c])=>c.end<t&&t-c.end<3);if(p.length){a=[p[p.length-1]];note='<span class="tag">только что закончился</span>'}}
let h='';const g=D.gt.filter(z=>t>=z.start&&t<=z.end);if(g.length)h+=`<div style="margin-bottom:6px"><span class="tag">эталон: ${g.map(z=>z.label==='POSITIVE'?'курение':z.label==='IGNORE'?'игнор':'не курение').join(', ')}</span></div>`;if(!a.length)h+='<span class="mut">Нет цикла под курсом. Циклы — на таймлайне.</span>';
a.forEach(([c,k])=>{h+=`<h4>цикл #${k} · ID${c.tid} ${note}</h4><div class="mut">${c.start.toFixed(1)}–${c.end.toFixed(1)} с · пауза ${fmt(c.hold)} с · рука ${c.hand==0?'L':c.hand==1?'R':'?'}</div>`
+meter('итог (score)',c.score)+meter('дешёвый пакет',c.score_cheap)+meter('фото-модель',c.photo)+meter('VLM «затяжка»',c.vlm)+meter('предмет (conf)',c.obj)
+(c.label?`<div style="margin-top:4px">метка: <span class="tag">${c.label}</span></div>`:'')});
document.getElementById('cur').innerHTML=h;document.getElementById('tm').textContent=` t=${t.toFixed(2)} / ${(off+dur).toFixed(1)} с`}
function evs(){document.getElementById('evs').innerHTML=D.events.length?D.events.map((e,i)=>`<div class="ev" data-t="${e.start}">ID${e.tid}: ${e.start.toFixed(1)}–${e.end.toFixed(1)} с <b>${fmt(e.conf)}</b><div class="mut">${e.note||''}</div></div>`).join(''):'<span class="mut">событий нет</span>';
document.querySelectorAll('.ev').forEach(el=>el.onclick=()=>{seek(+el.dataset.t-.5);v.play()})}
function errs(){document.getElementById('errh').innerHTML=D.errors.length?'<h4 style="margin-top:10px">Ошибки ('+D.errors.length+')</h4>':'';
document.getElementById('errs').innerHTML=D.errors.map(e=>`<div class="er" data-t="${e.start}"><b>${e.kind}</b> ${e.start.toFixed(1)} с${e.conf!=null?' · '+e.conf.toFixed(2):''}<div class="mut">${e.text||''}</div></div>`).join('');
document.querySelectorAll('.er').forEach(el=>el.onclick=()=>{seek(+el.dataset.t-1.5);v.play()})}
function jumpErr(dir){const t=now(),s=D.errors.map(e=>e.start).sort((a,b)=>a-b);const x=dir>0?s.find(z=>z-1.5>t+.2):[...s].reverse().find(z=>z-1.5<t-.3);if(x!=null)seek(Math.max(0,x-1.5))}
/* рамки людей поверх видео */
const ov=document.getElementById('ov'),ox=ov.getContext('2d');
function boxAt(tid,t){const p=D.tracks[tid];if(!p||!p.length)return null;let lo=0,hi=p.length-1;while(lo<hi){const m=(lo+hi)>>1;if(p[m][0]<t)lo=m+1;else hi=m}let i=lo;if(i>0&&Math.abs(p[i-1][0]-t)<Math.abs(p[i][0]-t))i--;return Math.abs(p[i][0]-t)<=1.0?p[i].slice(1):null}
function sizeOv(){const r=v.getBoundingClientRect(),d=window.devicePixelRatio||1;ov.width=r.width*d;ov.height=r.height*d;ov.style.width=r.width+'px';ov.style.height=r.height+'px';ox.setTransform(d,0,0,d,0,0)}
function scr(b){const r=v.getBoundingClientRect(),vw=v.videoWidth||16,vh=v.videoHeight||9,sc=Math.min(r.width/vw,r.height/vh),ox_=(r.width-vw*sc)/2,oy=(r.height-vh*sc)/2,sw=D.src_w||vw,k=vw*sc/sw;return[ox_+b[0]*k,oy+b[1]*k,ox_+b[2]*k,oy+b[3]*k]}
function drawOv(){const r=v.getBoundingClientRect();ox.clearRect(0,0,r.width,r.height);if(!showBox||!D.src_w)return;const t=now();ox.font='12px system-ui';
 const hot=new Set(D.events.filter(e=>t>=e.start&&t<=e.end).map(e=>String(e.tid)));
 Object.keys(D.tracks).forEach(tid=>{const b=boxAt(tid,t);if(!b)return;const q=scr(b),on=hot.has(tid);ox.strokeStyle=on?'#ff5a4d':'rgba(255,255,255,.7)';ox.lineWidth=on?3:1.2;ox.strokeRect(q[0],q[1],q[2]-q[0],q[3]-q[1]);ox.fillStyle=on?'#ff5a4d':'rgba(0,0,0,.6)';ox.fillRect(q[0],q[1]-15,on?78:34,15);ox.fillStyle='#fff';ox.fillText('ID '+tid+(on?' · курит?':''),q[0]+3,q[1]-3)});
 D.gt.filter(g=>g.box&&g.label==='POSITIVE'&&t>=g.start&&t<=g.end).forEach(g=>{const q=scr(g.box);ox.strokeStyle='#2fb36b';ox.lineWidth=2;ox.setLineDash([5,3]);ox.strokeRect(q[0],q[1],q[2]-q[0],q[3]-q[1]);ox.setLineDash([])})}
function seek(t){v.currentTime=Math.max(0,Math.min(dur,t-off))}
function step(n){v.pause();seek(now()+n/fps)}
function jump(dir){const t=now(),s=D.cycles.map(c=>c.start).sort((a,b)=>a-b);let x=dir>0?s.find(z=>z>t+.05):[...s].reverse().find(z=>z<t-.6);if(x!=null)seek(Math.max(0,x-.4))}
function setLoop(k){loop=(loop===k)?null:k;document.getElementById('lp').classList.toggle('on',loop!==null);if(loop!==null){seek(D.cycles[loop].start-.3);v.play()}}
function spd(d){si=Math.max(0,Math.min(speeds.length-1,si+d));v.playbackRate=speeds[si];document.getElementById('sp').textContent=speeds[si]+'×'}
cv.addEventListener('click',e=>{const r=cv.getBoundingClientRect();seek(T(e.clientX-r.left))});
cv.addEventListener('dblclick',e=>{const r=cv.getBoundingClientRect(),t=T(e.clientX-r.left),y=e.clientY-r.top;const k=D.cycles.findIndex(c=>t>=c.start&&t<=c.end&&ry(tids.indexOf(c.tid))<=y&&y<=ry(tids.indexOf(c.tid)+1));if(k>=0)setLoop(k)});
document.getElementById('pp').onclick=()=>v.paused?v.play():v.pause();document.getElementById('bk').onclick=()=>step(-1);document.getElementById('fw').onclick=()=>step(1);
document.getElementById('pc').onclick=()=>jump(-1);document.getElementById('nc').onclick=()=>jump(1);document.getElementById('sl').onclick=()=>spd(-1);document.getElementById('sf').onclick=()=>spd(1);
document.getElementById('lp').onclick=()=>{const t=now();const k=D.cycles.findIndex(c=>t>=c.start-.3&&t<=c.end);if(loop!==null)setLoop(loop);else if(k>=0)setLoop(k)};
document.body.addEventListener('keydown',e=>{const k=e.key;if(k===' '){e.preventDefault();v.paused?v.play():v.pause()}else if(k==='ArrowLeft'){e.preventDefault();e.shiftKey?seek(now()-1):step(-1)}
else if(k==='ArrowRight'){e.preventDefault();e.shiftKey?seek(now()+1):step(1)}else if(k===','){jump(-1)}else if(k==='.'){jump(1)}else if(k==='l'||k==='L'){document.getElementById('lp').click()}else if(k==='['){spd(-1)}else if(k===']'){spd(1)}else if(k==='e'){jumpErr(1)}else if(k==='E'){jumpErr(-1)}});
v.addEventListener('timeupdate',()=>{if(loop!==null){const c=D.cycles[loop];if(now()>c.end+.3)seek(c.start-.3)}});
function tick(){draw();drawOv();panel();requestAnimationFrame(tick)}
document.getElementById('bx').onclick=function(){showBox=!showBox;this.classList.toggle('on',showBox)};
document.getElementById('ne').onclick=()=>jumpErr(1);document.getElementById('pe').onclick=()=>jumpErr(-1);
window.addEventListener('resize',()=>{size();sizeOv()});size();sizeOv();errs();evs();v.addEventListener('loadedmetadata',()=>{if(isFinite(v.duration)&&v.duration>0)dur=v.duration;sizeOv()},{once:true});tick();
</script></body></html>"""


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def build_data(cycles: pd.DataFrame, events: pd.DataFrame, duration: float, fps: float, offset: float = 0.0, *, gt: list | None = None, errors: list | None = None,
               tracks: dict | None = None, src_size: tuple[int, int] = (0, 0)) -> dict:
    """Данные для компонента: циклы с выходами моделей и события. Недостающие колонки допустимы (показываются «—»). `offset` — начало окна в секундах исходного видео:
    время в mp4 с разметкой отсчитывается от начала окна, а циклы и события заданы в абсолютном времени видео."""
    cyc = []
    if len(cycles):
        for r in cycles.sort_values("start").itertuples():
            g = lambda n: _num(getattr(r, n, None))   # noqa: E731
            cyc.append(dict(tid=int(r.tid), start=float(r.start), end=float(getattr(r, "end", r.start + 1.0)), hold=g("hold"), hand=int(getattr(r, "hand", -1)),
                            score=g("score"), score_cheap=g("score_cheap"), photo=g("photo_p_mean"), vlm=g("vlm_yesno"),
                            obj=g("obj_any_max_conf"), label=str(getattr(r, "label", "") or "") if str(getattr(r, "label", "") or "") not in ("nan", "None") else ""))
    ev = []
    if len(events):
        for r in events.itertuples():
            ev.append(dict(tid=int(r.person_track_id), start=float(r.start_sec), end=float(r.end_sec), conf=_num(r.confidence), note=str(getattr(r, "label", ""))))
    tids = sorted({c["tid"] for c in cyc} | {e["tid"] for e in ev})
    return dict(cycles=cyc, events=ev, duration=float(duration), fps=float(fps) or 15.0, tids=tids, offset=float(offset), gt=gt or [], errors=errors or [], tracks=tracks or {},
                src_w=int(src_size[0]), src_h=int(src_size[1]))


def player_html(video: str | Path | None, data: dict, timeline_height: int | None = None, video_uri: str | None = None) -> tuple[str, str | None]:
    """(HTML, предупреждение). Видео вшивается как data-URI, если оно меньше MAX_EMBED_MB; иначе плеер без картинки + предупреждение."""
    warn = None
    src = f'src="{video_uri}"' if video_uri else ""
    if video and not video_uri and Path(video).exists():
        mb = Path(video).stat().st_size / 1e6
        if mb <= MAX_EMBED_MB:
            src = f'src="data:video/mp4;base64,{base64.b64encode(Path(video).read_bytes()).decode()}"'
        else:
            warn = f"видео {mb:.0f} МБ больше {MAX_EMBED_MB:.0f} МБ — в плеер не вшито; используйте стандартный проигрыватель ниже"
    lanes = (1 if data.get("gt") else 0) + (1 if data.get("errors") else 0)
    h = timeline_height or (18 + 22 * (lanes + max(1, len(data["tids"]))) + 6)
    html = _HTML.replace("__SRC__", src).replace("__H__", str(h)).replace("__DATA__", json.dumps(data, ensure_ascii=False))
    return html, warn
