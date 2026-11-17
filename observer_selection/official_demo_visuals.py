"""Measured prediction plots and offline skeleton video; never fabricate poses."""
from pathlib import Path
import json
import numpy as np

PARENTS=(-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19)


def render_demo(output,pred,truth,ego,observer_video,disclaimer):
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output);N=len(pred)
    if pred.shape!=truth.shape or pred.shape!=ego.shape or pred.shape[1:]!=(22,3):
        raise ValueError('Expected aligned [T,22,3] joints')
    if not all(np.isfinite(x).all() for x in (pred,truth,ego)):raise ValueError('Nonfinite joints')
    errors=np.linalg.norm(pred-truth,axis=-1)*1000
    ego_errors=np.linalg.norm(ego-truth,axis=-1)*1000
    np.savetxt(out/'per_frame_errors.csv',np.c_[np.arange(2,N+2),np.arange(N)/20,errors.mean(1),ego_errors.mean(1)],
        delimiter=',',header='raw_frame_id,seconds_from_first_prediction,fusion_dinov2_mpjpe_mm,egonet_mpjpe_mm',comments='')
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(np.arange(N)/20,errors.mean(1),label='Pretrained fusion + DINOv2')
    ax.plot(np.arange(N)/20,ego_errors.mean(1),label='Official EgoNet component',alpha=.8)
    ax.set(xlabel='Time from first prediction (s)',ylabel='MPJPE (mm)',title='Single-pair demo: DINOv2 substitution')
    ax.legend();fig.text(.02,.01,'Synthetic GT-derived tracker / floor / body shape. Not full-paper reproduction.',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/'error_curve.png',dpi=300);fig.savefig(out/'error_curve.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,5),subplot_kw={'projection':'3d'})
    for ax,t in zip(axes,[0,N//2,N-1]):
        origin=truth[t,0]
        for joints,color,label in ((truth[t]-origin,'#e59827','Reference'),(pred[t]-origin,'#2878c8','Prediction')):
            for j,parent in enumerate(PARENTS):
                if parent>=0:ax.plot(*joints[[parent,j]].T,color=color,lw=2)
            ax.scatter(*joints.T,s=8,color=color,label=label)
        ax.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1.2,1.2),title=f'Raw frame {t+2}',xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
        ax.set_box_aspect((1,1,1.2))
    axes[0].legend();fig.suptitle('Official weights + DINOv2 / synthetic tracker demo')
    fig.tight_layout();fig.savefig(out/'pose_comparison.png',dpi=240);plt.close(fig)
    cap=cv2.VideoCapture(str(observer_video));cap.read() # raw frame 1 omitted by official temporal input
    writer=cv2.VideoWriter(str(out/'demo.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1280,720))
    if not cap.isOpened() or not writer.isOpened():raise RuntimeError('Cannot read/write demo video')
    try:
        for t in range(N):
            ok,image=cap.read()
            if not ok:raise ValueError('Observer video shorter than aligned predictions')
            frame=np.full((720,1280,3),245,np.uint8)
            frame[110:590,10:490]=cv2.resize(image,(480,480))
            cv2.putText(frame,'Pretrained EgoExoMoCap + DINOv2 (substitution)',(20,32),0,.8,(30,30,30),2)
            cv2.putText(frame,'Synthetic tracker / GT floor and shape. Not exact paper reproduction.',(20,65),0,.55,(30,30,30),1)
            cv2.putText(frame,f'Raw frame {t+2} | MPJPE {errors[t].mean():.1f} mm',(20,665),0,.65,(30,30,30),2)
            for horizontal,cx,label in ((0,690,'X-Z view'),(1,1080,'Y-Z view')):
                cv2.putText(frame,label,(cx-90,110),0,.7,(30,30,30),2)
                for joints,color in ((truth[t],(40,150,230)),(pred[t],(220,110,40))):
                    local=joints-truth[t,0]
                    points=np.stack([cx+local[:,horizontal]*150,370-local[:,2]*150],axis=-1).round().astype(int)
                    for j,parent in enumerate(PARENTS):
                        if parent>=0:cv2.line(frame,tuple(points[parent]),tuple(points[j]),color,3,cv2.LINE_AA)
                    for pt in points:cv2.circle(frame,tuple(pt),4,color,-1)
            cv2.putText(frame,'Blue: prediction   Orange: fitted reference',(520,620),0,.6,(30,30,30),1)
            writer.write(frame)
    finally:cap.release();writer.release()
    # Browser-independent interactive canvas; no WebGL/CDN dependencies.
    payload=json.dumps({'pred':pred.tolist(),'gt':truth.tolist(),'parents':PARENTS},allow_nan=False).replace('<','\\u003c')
    html='''<!doctype html><meta charset="utf-8"><title>EgoExoMoCap DINOv2 demo</title>
<style>body{font:16px system-ui;max-width:1100px;margin:24px auto;background:#f4f1eb}canvas{background:white}input{width:330px}</style>
<h1>Official pretrained weights + DINOv2 substitution</h1>
<p>Synthetic GT-derived head/wrist, floor and shape. Single pair, not exact paper reproduction.</p>
<p>Blue: prediction. Orange: fitted reference. Both centered at the reference pelvis for display only.</p>
<button id="play">Play / Pause</button> Frame <input id="frame" type="range" min="0" value="0"> Rotate <input id="yaw" type="range" min="-180" max="180" value="30">
<p id="info"></p><canvas id="pose" width="1000" height="600"></canvas>
<p><a href="error_curve.png">Error curve</a> · <a href="pose_comparison.png">Pose figure</a> · <a href="demo.mp4">Video</a></p>
<script>const d=PAYLOAD, f=document.querySelector('#frame'), y=document.querySelector('#yaw'), c=document.querySelector('#pose').getContext('2d');f.max=d.pred.length-1;let playing=false;
function draw(){let t=+f.value,a=+y.value*Math.PI/180,origin=d.gt[t][0];c.clearRect(0,0,1000,600);document.querySelector('#info').textContent='Raw frame '+(t+2);
for(let [arr,color] of [[d.gt[t],'#e59827'],[d.pred[t],'#2878c8']]){let p=arr.map(v=>{let x=v[0]-origin[0],y=v[1]-origin[1],z=v[2]-origin[2];return [500+200*(x*Math.cos(a)+y*Math.sin(a)),300-200*z]});c.strokeStyle=color;c.fillStyle=color;c.lineWidth=3;for(let j=0;j<22;j++){let k=d.parents[j];if(k>=0){c.beginPath();c.moveTo(...p[k]);c.lineTo(...p[j]);c.stroke()}c.beginPath();c.arc(...p[j],4,0,7);c.fill()}}}
f.oninput=draw;y.oninput=draw;document.querySelector('#play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){f.value=(+f.value+1)%d.pred.length;draw()}},50);draw();</script>'''.replace('PAYLOAD',payload)
    (out/'viewer.html').write_text(html,encoding='utf-8')
