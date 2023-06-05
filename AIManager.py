import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable as SM
import numpy as np
import os, time, json, re

import torch

#TODO class ui():

class Plots():
    def __init__(self, root_dir='logs', config='plotcfg'):
        self.root_dir = root_dir
        self.logs = self.root_dir+'/logs/'
        self.plots = self.root_dir+'/plots/'
        
        self.config = self.root_dir+'/'+config
        self.pyplot_config = {}
        
        self.log_files = {}
        for i in sort_txt(os.listdir(self.logs)):
            i = i.replace('.npy', '').split('_', 3)
            if i[0] == 'data':
                self.log_files['data'] = self.log_files.get('data', {})
                self.log_files['data'][i[1]] = self.log_files['data'].get(i[1], [])
                self.log_files['data'][i[1]].append(i[2:])
            elif i[0] == 'stat':
                self.log_files['stat'] = self.log_files.get('stat', {})
                self.log_files['stat'][i[1]] = self.log_files['stat'].get(i[1], [])
                self.log_files['stat'][i[1]].append(i[2:])
        
        plt.close('all')
        self.plot_data = {}
        self.pfig = plt.figure()
        self.pax = plt.axes()
        
        self.image_data = {}
        self.arr_fill = 0
        self.dt = np.float16
        self.dpi = 50
        self.bg_cmap = 'gray'
        self.img_cmap = 'viridis'
        self.image_arr = np.full((480, 640), self.arr_fill, dtype=self.dt)
        self.image_arr = SM(cmap=self.bg_cmap).to_rgba(self.image_arr)
        self.imfig = plt.figure(dpi=self.dpi)
        self.imax = plt.axes()
        self.axim = self.imax.imshow(self.image_arr)
    
    def add_stat(self, key):
        self.plot_data[key]={}
        
        for i in self.log_files['stat'][key]:
            data = np.loadtxt(self.logs+'stat_'+key+'_'+'_'.join(i))
                
        if len(data.shape) == 1:
            data = np.array([data])
                
        self.plot_data[key]['step'] = data.T[0].astype(np.float16)
        self.plot_data[key]['data'] = data.T[1].astype(np.float32)
    
    def add_data(self, key):
        self.plot_data[key]={}
        
        data = []
        step = []
        for i in self.log_files['data'][key]:
            arr = np.load(self.logs+'data_'+key+'_'+'_'.join(i)+'.npy')
            arr = arr.flatten()
            data.append(arr)
            step.append(i[0])
                
        step = np.array(step, dtype=np.float16)
        data = np.array(data, dtype=np.float32)
        self.plot_data[key]['step'] = step
        self.plot_data[key]['data'] = data.T
    
    def add_image_data(self, key, posx, posy, scale=1, rot=0, mndx=None, cmap=None):
        if mndx == None:
            mndx=''
        else:
            mndx = ''.join(['.'+str(i) for i in mndx])
            
        ndx = 0
        while self.image_data.get(str(ndx)+'.'+key+mndx):
            ndx+=1
        
        if cmap == None:
            cmap = self.img_cmap
            
        self.image_data[str(ndx)+'.'+key+mndx] = {'posx':posx,
                                                  'posy':posy,
                                                  'scale':scale,
                                                  'rot':rot,
                                                  'cmap': cmap}
    
    def update_image_transforms(self, key, posx=None, posy=None, scale=None, rot=None):
        if self.image_data.get(key):
            if posx != None:
                self.image_data[key]['posx'] = posx
            if posy != None:
                self.image_data[key]['posy'] = posy
            if scale != None:
                self.image_data[key]['scale'] = scale
            if rot != None:
                self.image_data[key]['rot'] = rot
        else:
            print("object doesn't exist. available keys are: {}".format(self.self.image_data.keys()))
    
    def blit_image_data(self, key, ndx=0):
        if self.image_data.get(key):
            mndx = None
            f_name = None
            mndx = [int(i) for i in key.split('.')[2:]]
            
            posx  = self.image_data[key]['posx']
            posy  = self.image_data[key]['posy']
            rot   = self.image_data[key]['rot']
            scale = self.image_data[key]['scale']
            cmap = self.image_data[key]['cmap']
            
            key = key.rsplit('.')[1]
            f_name = '_'.join(self.log_files['data'][key][ndx])
            f_name = 'data_'+key+'_'+f_name+'.npy'
            
            if f_name != None:
                data = np.load(self.logs+f_name)
                if mndx != []:
                    data = rec_get(data, mndx)
                data = np.kron(data, np.ones((scale, scale)))
                
                crds = np.indices((data.shape[0], data.shape[1]))
                if rot == 0:
                    crdsy = crds[0]
                    crdsx = crds[1]
                elif rot == 90:
                    crds_ = np.full((data.shape[0], data.shape[1]), data.shape[0]-1)
                    crdsy = crds[1]
                    crdsx = (crds_ - crds[0])
                elif rot == 180:
                    crds_ = np.full((data.shape[0], data.shape[1]), data.shape[0]-1)
                    crdsy = (crds_ - crds[0])
                    crds_ = np.full((data.shape[0], data.shape[1]), data.shape[1]-1)
                    crdsx = (crds_ - crds[1])
                elif rot == 270 or rot == -90:
                    crds_ = np.full((data.shape[0], data.shape[1]), data.shape[0]-1)
                    crdsy = (crds_ - crds[1])
                    crdsx = crds[0]
                    
                data = SM(cmap=cmap).to_rgba(data)
                self.image_arr[posy:posy+data.shape[0], posx:posx+data.shape[1]] = data
        else:
            print("object doesn't exist. available keys are: {}".format(self.image_data.keys()))
    
    def resize_image(self, width=None, height=None):
        if width != None and height != None:
            self.image_arr = np.full((height, width), self.arr_fill, dtype=self.dt)
        elif height != None:
            self.image_arr = np.full((height, self.image_arr.shape[1]), self.arr_fill, dtype=self.dt)
        elif width != None:
            self.image_arr = np.full((self.image_arr.shape[0], width), self.arr_fill, dtype=self.dt)
    
    def render_image(self, ndx=None):
        self.image_arr = np.full((self.image_arr.shape[0], self.image_arr.shape[1]), self.arr_fill, dtype=self.dt)
        self.image_arr = SM(cmap=self.bg_cmap).to_rgba(self.image_arr)
        for k in self.image_data.keys():
            self.blit_image_data(k, ndx)
        self.imax.clear()
        self.axim = self.imax.imshow(self.image_arr)
        self.imfig.canvas.draw()
    
    def render_image_sequence(self):
        steps = []
        ndx = []
        end = []
        self.image_arr = np.full((self.image_arr.shape[0], self.image_arr.shape[1]), self.arr_fill, dtype=self.dt)
        self.image_arr = SM(cmap=self.bg_cmap).to_rgba(self.image_arr)
        name = []
        num = []
        date = -1
        for k in self.image_data.keys():
            lk = k.split('.')[1]
            strt = self.log_files['data'][lk][0][0]
            nxt = self.log_files['data'][lk][1][0]
            steps.append(int(nxt) - int(strt))
            ndx.append(1)
            end.append(False)
            if lk not in name:
                name.append(lk)
            if self.log_files['data'][lk][0][0] not in num:
                num.append(self.log_files['data'][lk][0][0])
            if int_time(self.log_files['data'][lk][0][1]) > date:
                date = int_time(self.log_files['data'][lk][0][1])
            self.blit_image_data(k, 0)
        self.imax.clear()
        self.axim = self.imax.imshow(self.image_arr)
        self.imfig.canvas.draw()
        f_name = self.plots+'_'.join(name)+'_'+'_'.join(num)+'_'+str_time(date)
        print("Saving: {}".format(f_name))
        self.imfig.savefig(f_name)  
         
        while np.all(end) != True:
            min_ = 1e10
            ndx_ = -1
            for i, k in enumerate(self.image_data.keys()):
                if steps[i]*ndx[i] < min_ and ndx[i] < len(self.log_files['data'][lk]):
                    min_ = steps[i]*ndx[i]
                    ndx_ = i
                    
            ndxc = ndx.copy()
            for i, k in enumerate(self.image_data.keys()):
                if steps[i]*ndxc[i] == steps[i]*ndxc[ndx_] and \
                    ndx[i] <  len(self.log_files['data'][lk]):
                    ndx[i] += 1
    
            for i, k in enumerate(self.image_data.keys()):
                if ndx[i] >= len(self.log_files['data'][lk]):
                    end[i] = True
            
                    
            num = []
            date = -1
            self.image_arr = np.full((self.image_arr.shape[0], self.image_arr.shape[1]), self.arr_fill, dtype=self.dt)
            self.image_arr = SM(cmap=self.bg_cmap).to_rgba(self.image_arr)
            for i, k in enumerate(self.image_data.keys()):
                lk = k.split('.')[1]
                if self.log_files['data'][lk][ndx[i]-1][0] not in num:
                    num.append(self.log_files['data'][lk][ndx[i]-1][0])
                if int_time(self.log_files['data'][lk][ndx[i]-1][1]) > date:
                    date = int_time(self.log_files['data'][lk][ndx[i]-1][1])
                self.blit_image_data(k, ndx[i]-1)
            self.imax.clear()
            self.axim = self.imax.imshow(self.image_arr)
            self.imfig.canvas.draw()
            f_name = self.plots+'_'.join(name)+'_'+'_'.join(num)+'_'+str_time(date)
            print("Saving: {}".format(f_name))
            self.imfig.savefig(f_name)


class AIManager(Plots):
    def __init__(self, root_dir='logs', fresh=False, run=None, batch_size=None):
        self.models = {}
        self.optimizers = {}
        self.stats = {}
                    
        self.step = 0
        
        self._step = -1
        self.update = False
        
        self.save_state_time = time.time()
        self.track_stat_time = time.time()
        self.track_data_time = time.time()
        
        self.session_strt_str = str_time(time.time())
        self.session_strt_int = int(time.time())
        
        self.batch_size = batch_size
        if batch_size == None:
            print("batch size required")
            exit()
        
        self.root_dir = create_folders(root_dir)
        if fresh == True:
            self.run = self.session_strt_str
            self.run_file = self.root_dir+'/last_run'
            with open(self.run_file, 'w') as f:
                f.write(self.run)
                
            self.model_state_sav = create_folders(self.root_dir+'/'+self.run+'/model_states/')
            self.logs = create_folders(self.root_dir+'/'+self.run+'/logs/')
            self.plots = create_folders(self.root_dir+'/'+self.run+'/plots/')
        elif fresh == False and run == None:
            self.run_file = self.root_dir+'/last_run'
            if os.path.isfile(self.run_file):
                with open(self.run_file, 'r') as f:
                    self.run = f.read()
            else:
                print("last run file not found, starting fresh")
                self.run = self.session_strt_str
                self.run_file = self.root_dir+'/last_run'
                with open(self.run_file, 'w') as f:
                    f.write(self.run)
                    
            self.model_state_sav = create_folders(self.root_dir+'/'+self.run+'/model_states/')
            self.logs = create_folders(self.root_dir+'/'+self.run+'/logs/')
            self.plots = create_folders(self.root_dir+'/'+self.run+'/plots/')
        elif fresh == False and run != None:
            self.run = run
            if not os.path.isdir(self.root_dir+'/'+self.run):
                print("run: {} not found starting fresh".format(self.run))
                self.run = self.session_strt_str
                self.run_file = self.root_dir+'/last_run'
                with open(self.run_file, 'w') as f:
                    f.write(self.run)
                    
            self.model_state_sav = create_folders(self.root_dir+'/'+self.run+'/model_states/')
            self.logs = create_folders(self.root_dir+'/'+self.run+'/logs/')
            self.plots = create_folders(self.root_dir+'/'+self.run+'/plots/')
        
        super(Plots, self).__init__()

    def add_optimizers(self, **optim):
        for k, v in optim.items():
            self.optimizers[k] = v
    
    def add_models(self, **models):
        for k, v in models.items():
            self.models[k] = v
    
    def load_state(self, state=None):
        def load_model_states(data):
            if data != None:
                for m in self.models.keys():
                    self.models[m].load_state_dict(data[m])
                    self.models[m].eval()
                
                for op in self.optimizers.keys():
                    self.optimizers[op].load_state_dict(data[op])
                    
                self.step = data['step_count']
                print("Done.")
                    
        def del_data_past_state(state):
            for i in os.listdir(self.logs):
                if i.startswith('data_'):
                    data_step = i.split('_')[2]
                    state_step = state.split('_')[1]
                    if int(data_step) > int(state_step):
                        os.remove(self.logs+i)
                        
        data = None
        
        if state == None:
            try:
                state = sort_txt(os.listdir(self.model_state_sav))[-1]
                print("Loading: {}".format(self.model_state_sav+state))
                data = torch.load(self.model_state_sav+state)
                load_model_states(data)
                #load_last_stats()
                del_data_past_state(state)
            except IndexError:
                print("No states available")
        else:
            try:
                print("Loading: {}".format(state))
                data = torch.load(state)
                load_model_states(data)
                #load_last_stats()
                del_data_past_state(state)
            except FileNotFoundError:
                print("State {} Not available".format(state))
    
    def save_state(self, interval, tme=False):
        # state f_name = session start
        
        tme_ = time.time()
        if tme == False and self.step % interval == 0:
            print("....Saving State....")
            
            f_name = self.model_state_sav+'state_'+str(self.step)+'_'+self.session_strt_str
            state_time = time.time()
        
            data = {}
            for k, v in self.models.items():
                data[k] = v.state_dict()
            for k, v in self.optimizers.items():
                data[k] = v.state_dict()
        
            data['session_start'] = self.session_strt_str
            data['state_time'] = str_time(state_time)
            data['step_count'] = self.step
            data['batch_size'] = self.batch_size
            
            torch.save(data, f_name)
            
            self._step = self.step
            self.update = True
            print("....Done....")
        
        elif tme == True and tme_ - self.save_state_time > interval:
            self.save_state_time = time.time()
            print("....Saving State....")
            
            f_name = self.model_state_sav+'state_'+str(self.step)+'_'+self.session_strt_str
            state_time = str_time(time.time())
        
            data = {}
            for k, v in self.models.items():
                data[k] = v.state_dict()
            for k, v in self.optimizers.items():
                data[k] = v.state_dict()
        
            data['session_start'] = self.session_strt_str
            data['state_time'] = state_time
            data['step_count'] = self.step
            data['batch_size'] = self.batch_size
            
            torch.save(data, f_name)
        
            self._step = self.step
            self.update = True
            print("....Done....")
            
        if self._step != self.step:
            self.update = False
    
    def track_stat(self, interval, key, value, tme=False):
        tme_ = time.time()
        if tme == False and self.step % interval == 0:
            self.stats[key] = self.stats.get(key, [])
            self.stats[key].append([self.step, float(value)])
        elif tme == True and tme_ - self.track_stat_time > interval:
            self.track_stat_time = time.time()
            self.stats[key] = self.stats.get(key, [])
            self.stats[key].append([self.step, float(value)])
            
        if self.update == True:
            f_name = self.logs+'stat_'+key+'_'+self.run
            data = self.stats.get(key)
            if data:
                with open(f_name, 'a') as f:
                    np.savetxt(f, data)
                    
                del self.stats[key]
    
    def track_data(self, interval, key, value, tme=False):
        # data f_name = current time
        tme_ = time.time()
        if tme == False and self.step % interval == 0:
            f_name = self.logs+'data_'+key+'_'+str(self.step)+'_'+str_time(time.time())+'.npy'
            np.save(f_name, value)
        elif tme == True and tme_ - self.track_data_time > interval:
            self.track_data_time = time.time()
            f_name = self.logs+'data_'+key+'_'+str(self.step)+'_'+str_time(time.time())+'.npy'
            np.save(f_name, value)
        
        if tme_ % interval != 0:
            self.track_stat_run_once = True
    
    def print_update(self, **extra):
        print("="*38)
        print("Stats:")
        for k, v in self.stats.items():
            print("{}: {:.4f}".format(k, self.stats[k][-1][-1]))
        
        print("-"*30)
        for k, v in extra.items():
            try:
                print("{}: {:.4f}".format(k, v))
            except TypeError:
                print("{}: {}".format(k, v))
                
            
        if int(self.session_strt_str.split('_')[-1]) == 0:
            tme = time.ctime(time.time()-self.session_strt_int - 3600).split(' ')[4]
        else:
            tme = time.ctime(time.time()-self.session_strt_int).split(' ')[4]
        print("Running time: {}, Global step: {}".format(tme, self.step))
        print("="*38)


#Utils
def create_folders(f_name):
    if not os.path.exists(f_name):
        os.makedirs(f_name)
    return f_name

def load_as_dict(f_name):
    dict_ = {}
    h = []
    with open(f_name, 'r') as f:
        h = f.readline().replace('\n', '').split(',')
        for line in f.readlines():
            items = line.replace('\n', '').split(',')
            for ndx in range(len(items)):
                try:
                    item = float(items[ndx])
                except ValueError:
                    item = items[ndx]
                    
                dict_[h[ndx]] = dict_.get(h[ndx], [])
                dict_[h[ndx]].append(item)
    return dict_

def save_as_csv(dict_, f_name, mode):
    head = list(dict_.keys())
    with open(f_name, mode) as f:
        f.write(','.join(head)+'\n')
        get = True
        ndx = 0
        while get == True:
            line = []
            for n, k in enumerate(head):
                try:
                    item = str(dict_[k][ndx])
                    line.append(item)
                    get = True
                except TypeError:
                    item = str(dict_[k])
                    dict_[k] = [item]
                    line.append(item)
                    get = True
                except IndexError:
                    item = ""
                    line.append(item)
                    get = False
            f.write(','.join(line)+'\n')
            ndx += 1

def str_time(ti):
    return '_'.join([str(i) for i in time.localtime(ti)])

def int_time(ts):
    return int(time.mktime(time.struct_time([int(i) for i in ts.split('_')])))

def sort_txt(lst, ndx=None):
    conv = lambda txt : int(txt) if txt.isdigit() else txt
    alnum = lambda key : [ conv(i) for i in re.findall('([a-z]+|[0-9]+)', key.lower())]
    sel = lambda key: [ alnum(key)[i] for i in ndx]
    if ndx == None:
        return sorted(lst, key=alnum)
    else:
        return sorted(lst, key=sel)

def rec_get(data, ndx_lst,ndx=0):
    if ndx < len(ndx_lst)-1:
        data = data[ndx_lst[ndx]]
        ndx += 1
        return rec_get(data, ndx_lst, ndx)
    elif ndx == len(ndx_lst)-1:
        return data[ndx_lst[ndx]]

def rec_set(data, ndx_lst, val, ndx=0):
    if ndx < len(ndx_lst)-1:
        data = data[ndx_lst[ndx]]
        ndx += 1
        rec_set(data, ndx_lst, ndx)
    elif ndx == len(ndx_lst)-1:
        data[ndx_lst[ndx]] = val

def sync_steps(steps):
    synced = [[0 for i in steps]]
    cnts = [1 for i in steps]
    inc = None
    while inc != -1:
        val = 1e10
        inc = -1
        #synced.append([steps[i][cnts[i]] for i in range(len(cnts))])
        synced.append([cnts[i] for i in range(len(cnts))])
        for i in range(len(steps)):
            if cnts[i] < len(steps[i])-1:
                if steps[i][cnts[i]] < val:
                    val = steps[i][cnts[i]]
                    inc = i
        cnts[inc] += 1
    return synced

if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from PIL import Image
    
    XOR = {'FF':{'dat':[0.0, 0.0],'label': 0.0},
           'FT':{'dat':[0.0, 1.0],'label': 1.0},
           'TT':{'dat':[1.0, 1.0],'label': 0.0},
           'TF':{'dat':[1.0, 0.0],'label': 1.0}}
    
    
    class xor(nn.Module):
        def __init__(self, hidden_dim=5):
            super(xor, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.Sigmoid(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            
        def forward(self, x):
            return self.net(x)

    def logic_gate_img(size, points):
        h = size[0]
        w = size[1]
        #pixel co-ordinate
        ndx = (np.indices((h, w)).T)/np.array((h, w))
        #pixel distance from corner
        nw = np.sqrt(sum((np.array((0, 0)) - ndx).T**2))
        ne = np.sqrt(sum((np.array((0, 1)) - ndx).T**2))
        se = np.sqrt(sum((np.array((1, 1)) - ndx).T**2))
        sw = np.sqrt(sum((np.array((1, 0)) - ndx).T**2))
        #distance multiplied be corner value
        nw = nw/np.max(nw) * np.full((h, w), points[0])
        ne = ne/np.max(ne) * np.full((h, w), points[1])
        se = se/np.max(sei) * np.full((h, w), points[2])
        sw = sw/np.max(sw) * np.full((h, w), points[3])
        #norm
        arr = nw+se+ne+sw
        arr = arr - np.min(arr)
        arr = arr/np.max(arr)
        #
        arr = np.full((3, w, h), arr).T
        return arr
    
    batch_size = 16
    hidden_dim=3
    learning_rate=0.05
    training_loops = 30_000
    
    model = xor(hidden_dim=hidden_dim)
    opt = optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    model.train()
    
    log = AIManager(root_dir='test_run', batch_size=batch_size)
    log.add_models(xor=model)
    log.add_optimizers(opt=opt)
    log.load_state()
    
    
    # 0 ->999
    for i in range(training_loops):
        batch = [XOR[list(XOR.keys())[np.random.randint(0, len(XOR))]] for _ in range(batch_size)]
        batch_dat = [batch[i]['dat'] for i in range(batch_size)]
        batch_label = [[batch[i]['label']] for i in range(batch_size)]
        batch_dat = torch.tensor((batch_dat), dtype=torch.float32)
        batch_label = torch.tensor((batch_label), dtype=torch.float32)
        
        
        train_out = model.forward(batch_dat)
        loss = loss_func(train_out, batch_label)
        opt.zero_grad()
        loss.backward()
        opt.step() 
        
        log.track_stat(10, 'loss', loss)
        
        if i % 250 == 0:
            with torch.no_grad():
                input_ = [XOR[i]['dat'] for i in XOR.keys()]
                input_ = torch.tensor(input_).float()
                test_out = model.forward(input_)
                log.print_update(nw=test_out[0],
                                 ne=test_out[1],
                                 se=test_out[2],
                                 sw=test_out[3],
                                 step=i)
        
        log.save_state(10_000)
        
        
        
        log.track_data(60, 'outputs', test_out)
        log.track_data(60, 'net.0.weight', model.state_dict()['net.0.weight'])
        log.track_data(60, 'net.2.weight', model.state_dict()['net.2.weight'])
        log.track_data(60, 'net.4.weight', model.state_dict()['net.4.weight'])
        
        
        log.step += 1
