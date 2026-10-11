import {Fragment} from 'react';
// Deliberately small Markdown subset, rendered as escaped React text. No HTML.
function inline(text:string){return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part,i)=>part.startsWith('**')?<strong key={i}>{part.slice(2,-2)}</strong>:part.startsWith('`')?<code key={i}>{part.slice(1,-1)}</code>:<Fragment key={i}>{part}</Fragment>);}
export default function AnswerText({text}:{text:string}){
 if(!/\*\*|`|^[-*]\s|^\d+[.)]\s|^#{1,3}\s/m.test(text))return <p className="answer">{text}</p>;
 const lines=text.split('\n');const groups:{kind:string;lines:string[]}[]=[];
 for(const line of lines){const kind=/^[-*]\s/.test(line)?'ul':/^\d+[.)]\s/.test(line)?'ol':'text';const last=groups.at(-1);if(last?.kind===kind)last.lines.push(line);else groups.push({kind,lines:[line]});}
 return <div className="answer">{groups.map((g,i)=>g.kind==='ul'?<ul key={i}>{g.lines.map((line,j)=><li key={j}>{inline(line.replace(/^[-*]\s+/,''))}</li>)}</ul>:g.kind==='ol'?<ol key={i}>{g.lines.map((line,j)=><li key={j}>{inline(line.replace(/^\d+[.)]\s+/,''))}</li>)}</ol>:<p key={i}>{g.lines.map((line,j)=><Fragment key={j}>{j>0 && <br/>}{inline(line.replace(/^#{1,3}\s+/,''))}</Fragment>)}</p>)}</div>;
}
