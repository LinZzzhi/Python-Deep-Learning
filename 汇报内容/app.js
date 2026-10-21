'use strict';
const toast = message => {const el=document.querySelector('#toast');el.textContent=message;el.classList.add('visible');clearTimeout(toast.timer);toast.timer=setTimeout(()=>el.classList.remove('visible'),4500);};
function connectDemo(demo,url){demo.dataset.url=url;demo.querySelector('iframe').src=url;demo.querySelector('.external').href=url;}
document.querySelectorAll('.demo[data-url]').forEach(demo=>connectDemo(demo,demo.dataset.url));
function clearExpanded(){document.querySelectorAll('.demo.expanded').forEach(el=>el.classList.remove('expanded'));document.body.classList.remove('demo-open');updateButtons();}
function updateButtons(){document.querySelectorAll('.fullscreen').forEach(button=>{const demo=button.closest('.demo');button.textContent=(document.fullscreenElement===demo||demo.classList.contains('expanded'))?'⛶ 退出全屏':'⛶ 全屏演示';});}
document.querySelectorAll('.demo').forEach(demo=>{
 demo.querySelector('.reload').addEventListener('click',()=>{if(demo.dataset.url){demo.querySelector('iframe').src=demo.dataset.url;toast('正在重新加载项目');}else toast('请通过一键启动文件连接本地服务');});
 demo.querySelector('.fullscreen').addEventListener('click',async()=>{if(document.fullscreenElement){await document.exitFullscreen();return;}if(demo.classList.contains('expanded')){clearExpanded();return;}try{await demo.requestFullscreen();}catch{demo.classList.add('expanded');document.body.classList.add('demo-open');updateButtons();}});
 demo.querySelector('.external').addEventListener('click',event=>{if(!demo.dataset.url){event.preventDefault();toast('请先通过一键启动文件启动本地服务');}});
});
document.addEventListener('fullscreenchange',updateButtons);
document.addEventListener('keydown',event=>{if(event.key==='Escape')clearExpanded();});
const progress=document.querySelector('.progress');
addEventListener('scroll',()=>{const max=document.documentElement.scrollHeight-innerHeight;progress.style.width=(max>0?scrollY/max*100:0)+'%';},{passive:true});
const observer=new IntersectionObserver(entries=>{entries.forEach(entry=>{if(entry.isIntersecting){document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('active',a.hash==='#'+entry.target.id));}});},{rootMargin:'-15% 0px -65% 0px'});
document.querySelectorAll('.project').forEach(el=>observer.observe(el));
async function loadRuntime(){try{const response=await fetch('/runtime.json',{cache:'no-store'});if(!response.ok)throw Error('unavailable');const state=await response.json();const demo=document.querySelector('#finder-demo');connectDemo(demo,state.finder);demo.querySelector('iframe').hidden=false;document.querySelector('#finder-notice').hidden=true;}catch{}}
loadRuntime();
