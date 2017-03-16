# coding: utf-8

import chainer
import chainer.links as L
import chainer.functions as F



class LSTM_JSA(chainer.Chain):
    # Joints, Speaks, Annotations
    def __init__(self, dim_j=72, dim_s = 2, dim_a=2, dim_h=40,
                 lj=1.0, ls=1.0, la=1.0, train=True):
        super(LSTM_JSA, self).__init__(
            i_j = L.Linear(dim_j, dim_h),
            i_s = L.Linear(dim_s, dim_h),
            i_a = L.Linear(dim_a, dim_h),
            h_1 = L.LSTM(dim_h, dim_h),
            h_2 = L.LSTM(dim_h, dim_h),
            o_j = L.Linear(dim_h, dim_j),
            o_s = L.Linear(dim_h, dim_s),
            o_a = L.Linear(dim_h, dim_a),
        )
        self.lj, self.ls, self.la = lj, ls, la
        self.train = train
        
    def __call__(self, jnt, spk, ann, jnt_t, spk_t, ann_t):
        jnt_ = self.o_j(self.h_2(self.h_1(self.i_j(jnt))))
        spk_ = self.o_s(self.h_2(self.h_1(self.i_s(spk))))
        ann_ = self.o_a(self.h_2(self.h_1(self.i_a(ann))))
        # train=Trueならlossを返す
        if self.train:
            loss_j = F.mean_squared_error(jnt_, jnt_t)
            loss_s = F.mean_squared_error(spk_, spk_t)
            loss_a = F.mean_squared_error(ann_, ann_t)
            self.loss = self.lj*loss_j+self.ls*loss_s+self.la*loss_a
            return self.loss, loss_j, loss_s, loss_a
        else:
            self.pred_j = jnt_
            self.pred_s = spk_            
            self.pred_a = ann_
            return self.pred_j, self.pred_s, self.pred_a 

    def reset_state(self):
        self.h_1.reset_state()
        self.h_2.reset_state()


class LSTM_JVS_ANN(chainer.Chain):
    # Joints, Speaks, Annotations
    def __init__(self, dim_j=72, dim_v=72, dim_s = 2, dim_a=2, dim_h=50,
                 ratio=0, lx=1.0, la=1.0, train=True):
        super(LSTM_JVS_ANN, self).__init__(
            i_x = L.Linear(dim_j+dim_v+dim_s, dim_h),
            i_a = L.Linear(dim_a, dim_h),
            h_1 = L.LSTM(dim_h, dim_h),
            h_2 = L.LSTM(dim_h, dim_h),
            o_x = L.Linear(dim_h, dim_j+dim_v+dim_s),
            o_a = L.Linear(dim_h, dim_a),
        )
        self.dim_j, self.dim_v, self.dim_s, self.dim_a  = dim_j, dim_v, dim_s, dim_a
        self.ratio = ratio
        self.lx, self.la = lx, la
        self.train = train

        
    def __call__(self, jnt, vel, spk, ann, jnt_t, vel_t, spk_t, ann_t, apart=False):     
        x = F.concat((jnt, vel, spk))
        x = self.i_x(x)
        x = self.h_1(F.dropout(x, ratio=self.ratio))
        x = self.h_2(F.dropout(x, ratio=self.ratio))
        x_ = self.o_x(x)

        a = self.i_a(ann)
        a = self.h_1(F.dropout(a, ratio=self.ratio))
        a = self.h_2(F.dropout(a, ratio=self.ratio))
        a_ = self.o_a(a)

        if self.train:
            x_t = F.concat((jnt_t, vel_t, spk_t))            
            loss_x = F.mean_squared_error(x_, x_t)
            loss_a = F.mean_squared_error(a_, ann_t)
            self.loss = self.lx*loss_x + self.la*loss_a

            loss_j = F.mean_squared_error(x_.data[0, :self.dim_j],
                                          x_t.data[0, :self.dim_j])
            loss_v = F.mean_squared_error(x_.data[0, self.dim_j:self.dim_j+self.dim_v],
                                          x_t.data[0, self.dim_j:self.dim_j+self.dim_v])
            loss_s = F.mean_squared_error(x_.data[0, self.dim_j+self.dim_v:],
                                          x_t.data[0, self.dim_j+self.dim_v:])       
            return self.loss, loss_j, loss_v, loss_s, loss_a
        else:
            #予測値を返す
            if apart:
                #joint, vel, speak, anno,分割して返す
                self.pred_j = x_.data[0, :self.dim_j]         
                self.pred_v = x_.data[0, self.dim_j:self.dim_j+self.dim_v]
                self.pred_s = x_.data[0, self.dim_j+self.dim_v:]
                self.pred_a = a_.data[0, :]    
                return self.pred_j, self.pred_v, self.pred_s, self.pred_a
            
            #joint, vel, speak, annoを合体したままで返す
            self.pred_x = x_
            self.pred_a = a_
            return self.pred_x, self.pred_a 
                
            
    def reset_state(self):
        self.h_1.reset_state()
        self.h_2.reset_state()        



class LSTM_JS_ANN(chainer.Chain):
    # Joints, Speaks, Annotations
    def __init__(self, dim_j=72, dim_s = 2, dim_a=2, dim_h=50,
                 ratio=0, lx=1.0, la=1.0, train=True):
        super(LSTM_JS_ANN, self).__init__(
            i_x = L.Linear(dim_j+dim_s, dim_h),
            i_a = L.Linear(dim_a, dim_h),
            h_1 = L.LSTM(dim_h, dim_h),
            h_2 = L.LSTM(dim_h, dim_h),
            o_x = L.Linear(dim_h, dim_j+dim_s),
            o_a = L.Linear(dim_h, dim_a),
        )
        self.dim_j, self.dim_s, self.dim_a  = dim_j, dim_s, dim_a
        self.ratio = ratio
        self.lx, self.la = lx, la
        self.train = train

        
    def __call__(self, jnt, spk, ann, jnt_t, spk_t, ann_t, apart=False):     
        x = F.concat((jnt, spk))
        x = self.i_x(x)
        x = self.h_1(F.dropout(x, ratio=self.ratio))
        x = self.h_2(F.dropout(x, ratio=self.ratio))
        x_ = self.o_x(x)

        a = self.i_a(ann)
        a = self.h_1(F.dropout(a, ratio=self.ratio))
        a = self.h_2(F.dropout(a, ratio=self.ratio))
        a_ = self.o_a(a)

        if self.train:
            x_t = F.concat((jnt_t, spk_t))            
            loss_x = F.mean_squared_error(x_, x_t)
            loss_a = F.mean_squared_error(a_, ann_t)
            self.loss = self.lx*loss_x + self.la*loss_a

            loss_j = F.mean_squared_error(x_.data[0, :self.dim_j],
                                          x_t.data[0, :self.dim_j])
            loss_s = F.mean_squared_error(x_.data[0, self.dim_j:],
                                          x_t.data[0, self.dim_j:])       
            return self.loss, loss_j, loss_s, loss_a
        else:
            #予測値を返す
            if apart:
                #joint, vel, speak, anno,分割して返す
                self.pred_j = x_.data[0, :self.dim_j]         
                self.pred_s = x_.data[0, self.dim_j:]
                self.pred_a = a_.data[0, :]    
                return self.pred_j, self.pred_s, self.pred_a
            
            #joint, vel, speak, annoを合体したままで返す
            self.pred_x = x_
            self.pred_a = a_
            return self.pred_x, self.pred_a 
                
            
    def reset_state(self):
        self.h_1.reset_state()
        self.h_2.reset_state()         

        
 
class LSTMModel(chainer.Chain):
    
    def __init__(self, dim_x=72, dim_a=2, dim_h=20):
        super(LSTMModel, self).__init__(
            i_x = L.Linear(dim_x+dim_a, dim_h),
            h_1 = L.LSTM(dim_h, dim_h),
            h_2 = L.LSTM(dim_h, dim_h),
            o_x = L.Linear(dim_h, dim_x+dim_a),
        )
        self.dim_x = dim_x
        self.dim_a = dim_a
        
    def __call__(self, x, x_t, train=True):
        x_ = self.o_x(self.h_2(self.h_1(self.i_x(x))))

        # train=Trueならlossを返す
        if train:
            self.loss = F.mean_squared_error(x_, x_t)
            #可能ならここでjoint, annotation別の誤差を返す?
            loss_x = F.mean_squared_error(x_.data[0,:self.dim_x], x_t[0,:self.dim_x])
            loss_a = F.mean_squared_error(x_.data[0,-self.dim_a:], x_t[0,-self.dim_a:])
            #print x_.data[self.dim_x:], x_t[self.dim_x:]
            return self.loss, loss_x, loss_a
        else:
            self.pred = x_
            return self.pred
 
    def reset_state(self):
        self.h_1.reset_state()
        self.h_2.reset_state()   


class LSTM_ANN(chainer.Chain):
    
    def __init__(self, dim_x=72, dim_a=2, dim_h=20, lx=1.0, la=1.0, ratio = 0, train=True):
        super(LSTM_ANN, self).__init__(
            i_x = L.Linear(dim_x, dim_h),
            i_a = L.Linear(dim_a, dim_h),
            h_1 = L.LSTM(dim_h, dim_h),
            h_2 = L.LSTM(dim_h, dim_h),
            o_x = L.Linear(dim_h, dim_x),
            o_a = L.Linear(dim_h, dim_a),
        )
        self.lx=lx
        self.la=la
        self.train = train
        self.ratio = ratio
        
    def __call__(self, x, a, x_t, a_t):

        x = self.i_x(x)
        x = self.h_1(F.dropout(x, ratio=self.ratio))
        x = self.h_2(F.dropout(x, ratio=self.ratio))
        x_ = self.o_x(x)

        a = self.i_a(a)
        a = self.h_1(F.dropout(a, ratio=self.ratio))
        a = self.h_2(F.dropout(a, ratio=self.ratio))
        a_ = self.o_a(a)

        # train=Trueならlossを返す
        if self.train:
            loss_x = F.mean_squared_error(x_, x_t)
            loss_a = F.mean_squared_error(a_, a_t)
            self.loss = self.lx*loss_x + self.la*loss_a
            return self.loss, loss_x, loss_a
        else:
            self.pred_x = x_
            self.pred_a = a_
            return self.pred_x, self.pred_a 
 
    def reset_state(self):
        self.h_1.reset_state()
        self.h_2.reset_state()        


class LSTM_ANN_L3(chainer.Chain):
    
    def __init__(self, dim_x=72, dim_a=2, dim_h=20, lx=1.0, la=1.0, train=True):
        super(LSTM_ANN_L3, self).__init__(
            i_x = L.Linear(dim_x, dim_h),
            i_a = L.Linear(dim_a, dim_h),
            h_1 = L.LSTM(dim_h, dim_h),
            h_2 = L.LSTM(dim_h, dim_h),
            h_3 = L.LSTM(dim_h, dim_h),
            o_x = L.Linear(dim_h, dim_x),
            o_a = L.Linear(dim_h, dim_a),
        )
        self.lx=lx
        self.la=la
        self.train = train
        
    def __call__(self, x, a, x_t, a_t):
        x_ = self.o_x(self.h_3(self.h_2(self.h_1(self.i_x(x)))))
        a_ = self.o_a(self.h_3(self.h_2(self.h_1(self.i_a(a)))))
        # train=Trueならlossを返す
        if self.train:
            loss_x = F.mean_squared_error(x_, x_t)
            loss_a = F.mean_squared_error(a_, a_t)
            self.loss = self.lx*loss_x + self.la*loss_a
            return self.loss, loss_x, loss_a
        else:
            self.pred_x = x_
            self.pred_a = a_
            return self.pred_x, self.pred_a 
 
    def reset_state(self):
        self.h_1.reset_state()
        self.h_2.reset_state()
        self.h_3.reset_state()


class LSTM_ANN_A3(chainer.Chain):
    
    def __init__(self, dim_x=72, dim_a=2, dim_h=20, lx=1.0, la=1.0, train=True):
        super(LSTM_ANN_A3, self).__init__(
            i_x = L.Linear(dim_x, dim_h),
            i_a = L.Linear(dim_a, dim_h),
            h_1 = L.LSTM(dim_h, dim_h),
            h_2 = L.LSTM(dim_h, dim_h),
            o_x = L.Linear(dim_h, dim_x),
            o_a = L.Linear(dim_h, dim_a),
        )
        self.lx=lx
        self.la=la
        self.train = train
        
    def __call__(self, x, a, x_t, a_t):
        x_ = self.o_x(self.h_2(self.h_1(self.i_x(x))))
        a_ = self.o_a(self.h_2(self.i_a(a)))
        # train=Trueならlossを返す
        if self.train:
            loss_x = F.mean_squared_error(x_, x_t)
            loss_a = F.mean_squared_error(a_, a_t)
            self.loss = self.lx*loss_x + self.la*loss_a
            return self.loss, loss_x, loss_a
        else:
            self.pred_x = x_
            self.pred_a = a_
            return self.pred_x, self.pred_a 
 
    def reset_state(self):
        self.h_1.reset_state()
        self.h_2.reset_state()        


class LSTM_ANN_L3_A3(chainer.Chain):
    
    def __init__(self, dim_x=72, dim_a=2, dim_h=20, lx=1.0, la=1.0, train=True):
        super(LSTM_ANN_L3_A3, self).__init__(
            i_x = L.Linear(dim_x, dim_h),
            i_a = L.Linear(dim_a, dim_h),
            h_1 = L.LSTM(dim_h, dim_h),
            h_2 = L.LSTM(dim_h, dim_h),
            h_3 = L.LSTM(dim_h, dim_h),
            o_x = L.Linear(dim_h, dim_x),
            o_a = L.Linear(dim_h, dim_a),
        )
        self.lx=lx
        self.la=la
        self.train = train
        
    def __call__(self, x, a, x_t, a_t):
        x_ = self.o_x(self.h_3(self.h_2(self.h_1(self.i_x(x)))))
        a_ = self.o_a(self.h_3(self.i_a(a)))
        # train=Trueならlossを返す
        if self.train:
            loss_x = F.mean_squared_error(x_, x_t)
            loss_a = F.mean_squared_error(a_, a_t)
            self.loss = self.lx*loss_x + self.la*loss_a
            return self.loss, loss_x, loss_a
        else:
            self.pred_x = x_
            self.pred_a = a_
            return self.pred_x, self.pred_a 
 
    def reset_state(self):
        self.h_1.reset_state()
        self.h_2.reset_state()
        self.h_3.reset_state()
