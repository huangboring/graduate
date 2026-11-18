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


def raw_joint_overlay(sequence, images, ids, frames, requester, fixed):
    import cv2
    import pickle
    from .fixed_camera import FixedCamera
    from .raw_prepare import fisheye_project, read_calibration

    frame_index = {frame:i for i,frame in enumerate(ids)}
    fixed_maps = {}
    if fixed:
        workplace = sequence/'colmap/workplace'
        with (workplace/'colmap_from_aria_transforms.pkl').open('rb') as stream:
            world_to_colmap = pickle.load(stream)['aria01']
        for source in sorted(fixed):
            camera = FixedCamera(workplace,source,world_to_colmap,[images[source][i] for i in ids])
            fixed_maps[source] = (cv2.convertMaps(*camera.maps,cv2.CV_32FC1) if camera.maps is not None
                                  else np.asarray(camera.size)/camera.output_size)

    def joints_for(source, frame):
        item = frames[frame_index[frame]]
        obs = item.observation if source==requester else item.packets[source]
        valid = np.asarray(obs.valid,bool) & (np.asarray(obs.confidence)>=.15) & np.isfinite(obs.xy).all(-1)
        points = obs.xy[valid]
        if not len(points):
            return points
        if source in fixed_maps:
            mapping = fixed_maps[source]
            if not isinstance(mapping,tuple):
                return points*mapping
            xmap,ymap = mapping
            xy = np.rint(points).astype(int)
            h,w = xmap.shape
            x,y = xy[(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)].T
            return np.column_stack((xmap[y,x],ymap[y,x]))
        calibration = sequence/'ego'/source/'calib'/f'{frame:05}.txt'
        intr,_ = read_calibration(calibration)
        points = points[((points>=0)&(points<1408)).all(-1)]
        native = np.column_stack((points[:,1],1407-points[:,0]))
        raw = fisheye_project(np.column_stack(((native-703.5)/704.,np.ones(len(native)))),intr)
        return np.column_stack((1407-raw[:,1],raw[:,0]))

    return joints_for


def _tile(cv2, canvas, image, x, y, width, height, title, bright, joints=None):
    h, w = image.shape[:2]
    scale = min(width/w,(height-28)/h)
    sw, sh = max(1,int(w*scale)), max(1,int(h*scale))
    image = cv2.resize(image,(sw,sh),interpolation=cv2.INTER_AREA)
    if joints is not None:
        for u,v in joints:
            if np.isfinite([u,v]).all() and 0<=u<w and 0<=v<h:
                centre=(int(u*sw/w),int(v*sh/h))
                cv2.circle(image,centre,4,(16,20,22),-1,cv2.LINE_AA)
                cv2.circle(image,centre,2,(95,237,109),-1,cv2.LINE_AA)
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
    yaw, elevation = np.radians([35.,18.])
    sideways = np.array([np.cos(yaw),-np.sin(yaw),0.])
    forward = np.array([np.sin(yaw),np.cos(yaw),0.])
    up = np.array([0.,0.,1.])
    vertical = up*np.cos(elevation)-forward*np.sin(elevation)
    depth_axis = forward*np.cos(elevation)+up*np.sin(elevation)

    def project(points):
        depth = points @ depth_axis
        scale = 145*4/(4+np.maximum(depth,-2.))
        screen = np.column_stack((points @ sideways,-(points @ vertical)))
        return np.rint(screen*scale[:,None]+[370,174]).astype(int),depth

    foot = float(relative[[10,11],2].min())
    for offset in np.linspace(-1.2,1.2,9):
        for segment in (((-1.2,offset,foot),(1.2,offset,foot)),
                        ((offset,-1.2,foot),(offset,1.2,foot))):
            endpoints,_ = project(np.asarray(segment))
            cv2.line(image,tuple(endpoints[0]),tuple(endpoints[1]),(65,56,50),1,cv2.LINE_AA)
    coords,depth = project(relative)
    for joint in np.argsort(-depth):
        parent = parents[joint]
        if parent>=0:
            cv2.line(image,tuple(coords[joint]),tuple(coords[parent]),(247,196,102),3,cv2.LINE_AA)
        cv2.circle(image,tuple(coords[joint]),4,(248,224,169),-1,cv2.LINE_AA)
    cv2.putText(image,'3D | perspective',(25,325),cv2.FONT_HERSHEY_SIMPLEX,.65,(220,220,220),1,cv2.LINE_AA)
    return image


def write_video(destination, rows, predictions, candidates, local, images, fps=20, joints_for=None):
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
            _tile(cv2,canvas,read(local),12,12,390,350,'LOCAL  '+local,True,
                  joints_for(local,frame) if joints_for else None)
            joints = predictions[index-1] if index else None
            title = f'RECONSTRUCTION  |  Frame {frame}'
            _tile(cv2,canvas,_skeleton(cv2,joints),414,12,854,350,title,True)
            tile_width = (width-12)//columns
            for i,source in enumerate(candidates):
                bright = source in selected
                status = 'INITIALIZE' if index==0 else ('SELECTED' if bright else 'NOT SELECTED')
                _tile(cv2,canvas,read(source),12+(i%columns)*tile_width,top+(i//columns)*tile_height,
                      tile_width-11,tile_height-10,f'{source}  |  {status}',bright,
                      joints_for(source,frame) if bright and joints_for else None)
            writer.write(canvas)
    finally:
        writer.release()
    if not temporary.is_file() or temporary.stat().st_size < 512:
        raise RuntimeError('MP4 export produced no video')
    temporary.replace(destination)


def export_outputs(output, history, learned, local, all_views, warmup, candidates, requester,
                   predictions, images, omitted_frames, random_matched=None, joints_for=None):
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
        writer.writerow(['policy','raw_frame','mpjpe_mm','requests','selected_ids','refresh_ids','triangulated_joints'])
        for label,rows in methods.items():
            for row in rows:
                writer.writerow([label,row['frame'],row['mpjpe_mm'],len(row['selected']),
                                 ' '.join(row['selected']),' '.join(row.get('refresh',())),row['triangulated_joints']])
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
        write_video(output/'multiview_selection.mp4',[warmup,*learned],predictions,candidates,requester,images,joints_for=joints_for)
