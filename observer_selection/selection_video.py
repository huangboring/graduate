"""Charts and a frame-aligned multi-view animation, exported after inference."""
import csv
import math
import numpy as np

from .preparation_cache import atomic_json


def image_paths(sequence, sources, ids):
    frames = {}
    for source in sources:
        base = sequence/'exo'/source/'images'
        if not base.is_dir():
            base = sequence/'ego'/source/'images/rgb'
        by_id = {int(p.stem):p for p in base.iterdir()
                 if p.suffix.lower() in ('.jpg','.jpeg','.png') and p.stem.isdecimal()}
        absent = set(ids)-by_id.keys()
        if absent:
            raise ValueError(f'{source} missing {len(absent)} image frames, first {min(absent)}')
        frames[source] = by_id
    return frames


def _tile(cv2, canvas, image, x, y, width, height, title, bright):
    h, w = image.shape[:2]
    scale = min(width/w,(height-28)/h)
    sw, sh = max(1,int(w*scale)), max(1,int(h*scale))
    image = cv2.resize(image,(sw,sh),interpolation=cv2.INTER_AREA)
    if not bright:
        image = (image*.28).astype('u1')
    px, py = x+(width-sw)//2, y+28+(height-28-sh)//2
    canvas[py:py+sh,px:px+sw] = image
    color = (135,220,128) if bright else (105,115,127)
    cv2.rectangle(canvas,(x,y),(x+width-1,y+height-1),color,2)
    cv2.putText(canvas,title,(x+8,y+19),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)


def _skeleton(cv2, joints):
    image = np.full((350,740,3), (31,26,23),dtype='u1')
    if joints is None:
        cv2.putText(image,'Initialization | reconstruction starts at frame 2',(70,180),
                    cv2.FONT_HERSHEY_SIMPLEX,.65,(230,232,235),1,cv2.LINE_AA)
        return image
    parents = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
    relative = joints-joints[0]
    for k, label in enumerate(('XZ','YZ')):
        center = np.array([185+k*370,185])
        coords = np.rint(relative[:,[k,2]]*[150,-150]+center).astype(int)
        for joint,parent in enumerate(parents):
            if parent>=0:
                cv2.line(image,tuple(coords[joint]),tuple(coords[parent]),(247,196,102),3,cv2.LINE_AA)
        for xy in coords:
            cv2.circle(image,tuple(xy),4,(248,224,169),-1,cv2.LINE_AA)
        cv2.putText(image,label,(30+k*370,325),cv2.FONT_HERSHEY_SIMPLEX,.7,(220,220,220),1,cv2.LINE_AA)
    return image


def write_video(destination, rows, predictions, candidates, local, images, fps=20):
    import cv2

    columns = min(5,len(candidates))
    width, top, tile_height = 1280, 375, 177
    height = top + math.ceil(len(candidates)/columns)*tile_height + 12
    width, height = width//2*2, (height+1)//2*2
    temporary = destination.with_name(destination.stem+'.partial.mp4')
    writer = cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():
        raise RuntimeError('OpenCV MP4 writer unavailable')
    try:
        for index,row in enumerate(rows):
            frame, selected = row['frame'], set(row['selected'])
            canvas = np.full((height,width,3),(19,17,16),np.uint8)
            def read(source):
                image = cv2.imdecode(np.fromfile(images[source][frame],dtype=np.uint8),cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f'Unreadable video frame: {images[source][frame]}')
                return image
            _tile(cv2,canvas,read(local),12,12,390,350,'LOCAL  '+local,True)
            joints = predictions[index-1] if index else None
            title = f'RECONSTRUCTION  |  Frame {frame}'
            _tile(cv2,canvas,_skeleton(cv2,joints),414,12,854,350,title,True)
            tile_width = (width-12)//columns
            for i,source in enumerate(candidates):
                bright = source in selected
                status = 'INITIALIZE' if index==0 else ('SELECTED' if bright else 'NOT SELECTED')
                _tile(cv2,canvas,read(source),12+(i%columns)*tile_width,top+(i//columns)*tile_height,
                      tile_width-11,tile_height-10,f'{source}  |  {status}',bright)
            writer.write(canvas)
    finally:
        writer.release()
    if not temporary.is_file() or temporary.stat().st_size < 512:
        raise RuntimeError('MP4 export produced no video')
    temporary.replace(destination)


def export_outputs(output, history, learned, local, all_views, warmup, candidates, requester,
                   predictions, images, omitted_frames, random_matched=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = {'Learned':learned,'Local only':local,'All sources':all_views}
    if random_matched is not None:
        methods['Random (matched count)'] = random_matched
    metrics = {label:{'mpjpe_mm':float(np.mean([row['mpjpe_mm'] for row in rows])),
             'mean_requests':float(np.mean([len(row['selected']) for row in rows])),
             'mean_triangulated_joints':float(np.mean([row['triangulated_joints'] for row in rows]))}
               for label,rows in methods.items()}
    atomic_json(output/'metrics.json',{'scope':'IN-SAMPLE training sequence',
        'initialization':warmup,'omitted_tail_frames':omitted_frames,'methods':metrics})
    atomic_json(output/'selection.json',{'requester':requester,'candidates':list(candidates),
        'frames':[warmup,*learned]})
    with (output/'frames.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['policy','raw_frame','mpjpe_mm','requests','selected_ids','triangulated_joints'])
        for label,rows in methods.items():
            for row in rows:
                writer.writerow([label,row['frame'],row['mpjpe_mm'],len(row['selected']),
                                 ' '.join(row['selected']),row['triangulated_joints']])
    plot = output/'figures'
    plot.mkdir(exist_ok=True)
    fig,axes = plt.subplots(2,2,figsize=(13,8),layout='constrained')
    fig.suptitle('EgoHumans | training sequence (in-sample)')
    for label,rows in methods.items():
        axes[0,0].plot([r['frame'] for r in rows],[r['mpjpe_mm'] for r in rows],label=label,lw=1)
        m=metrics[label]
        axes[0,1].scatter(m['mean_requests'],m['mpjpe_mm'],label=label,s=65)
    axes[0,0].set(xlabel='Raw frame',ylabel='MPJPE (mm)')
    axes[0,0].legend()
    axes[0,1].set(xlabel='Mean external requests / frame',ylabel='Mean MPJPE (mm)')
    axes[0,1].legend()
    roster = np.array([[source in r['selected'] for r in [warmup,*learned]] for source in candidates])
    axes[1,0].imshow(roster,aspect='auto',interpolation='nearest',cmap='Greens',vmin=0,vmax=1,
                    extent=(warmup['frame']-.5,learned[-1]['frame']+.5,len(candidates)-.5,-.5))
    axes[1,0].set(xlabel='Raw frame | first frame is full initialization',yticks=range(len(candidates)),
                  yticklabels=candidates,title='Learned policy selection')
    axes[1,1].plot([h['epoch'] for h in history],[h['mpjpe_mm'] for h in history],marker='o')
    axes[1,1].set(xlabel='Epoch',ylabel='Training MPJPE (mm)')
    fig.savefig(plot/'overview.png',dpi=180)
    fig.savefig(plot/'overview.pdf')
    plt.close(fig)
    if images:
        write_video(output/'multiview_selection.mp4',[warmup,*learned],predictions,candidates,requester,images)
