# -*- coding: utf-8 -*-
import torch
import torch.nn
import torch.nn.functional as F


class PatchMatch(object):
    def __init__(self, image_src, image_dest, patch_size, initial_NNF=None, device=None):
        
        #check that patch size is odd
        assert patch_size%2==1
        
        
        if(device is None):
            self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device=device
        print(self.device , 'is being used for PatchMatch')
            
        
        m = torch.nn.ReflectionPad2d(patch_size // 2).to(self.device)
        self.image_src = m(image_src)[0].detach().clone().to(self.device)  # c x h x w
        self.image_dest = m(image_dest)[0].detach().clone().to(self.device)  # c x h x w

        self.patch_size = patch_size
        self.src_h = self.image_src.size(1) - patch_size + 1
        self.src_w = self.image_src.size(2) - patch_size + 1
        self.dest_h = self.image_dest.size(1) - patch_size + 1
        self.dest_w = self.image_dest.size(2) - patch_size + 1

        if initial_NNF is None:
            a = torch.randint(0, self.dest_h-1, size=(self.src_h, self.src_w, 1)).to(device)  # h x w x 1
            b = torch.randint(0, self.dest_w-1, size=(self.src_h, self.src_w, 1)).to(device)  # h x w x 1
            self.NNF = torch.cat([a, b], dim=2).to(self.device)  # h x w x 2        
        else:
            self.NNF = initial_NNF
        
        self.init_dist()


    @staticmethod
    def dist(patch1, patch2, patch3, patch4):
        print('unused')
        return torch.sum((patch1 - patch2).pow(2)) + torch.sum((patch3 - patch4).pow(2))

    @staticmethod
    def calc_dist_parallel(img_src,img_dst,patch_size,NNF,src_h,src_w, device):
        arr = torch.zeros(size=(src_h,src_w)).to(device)  # h x w
        #padding NNF to match the reflection size
        p3d = (0,0,patch_size//2,patch_size//2,patch_size//2,patch_size//2)
        nnf_padded=F.pad(NNF,p3d,"constant",0)
        
        #distance img_src and img_dst
        for offset_i in reversed(range(patch_size)):
            for offset_j in reversed(range(patch_size)):
                
                #shift dest_image and then slice it to NND size
                dest_shifted=torch.roll(img_dst,shifts=(offset_i-patch_size//2,offset_j-patch_size//2),dims=(1,2))
                dest_shifted=dest_shifted[:,nnf_padded[:,:, 0]+patch_size//2, nnf_padded[:,:, 1]+patch_size//2].view(dest_shifted.size())
                dest_shifted=dest_shifted[:,patch_size//2:dest_shifted.size(1)-patch_size//2,patch_size//2:dest_shifted.size(2)-patch_size//2]
                
                
                #shift image_src, apply padded NNF, and then slice it to NND size
                src_shifted=torch.roll(img_src,shifts=(offset_i-patch_size//2,offset_j-patch_size//2),dims=(1,2))
                
                src_NNF_shifted=src_shifted
                src_NNF_shifted=src_NNF_shifted[:,patch_size//2:src_NNF_shifted.size(1)-patch_size//2,patch_size//2:src_NNF_shifted.size(2)-patch_size//2]
                arr=arr+torch.sum((dest_shifted-src_NNF_shifted).pow(2),dim=0)
                #print(torch.sum((dest_shifted-src_NNF_shifted).pow(2),dim=0)[0][0])
        
        return arr
    
    
    def init_dist(self,):            
        self.NND=self.calc_dist_parallel(self.image_src,self.image_dest,self.patch_size,self.NNF,self.src_h,self.src_w, self.device)
            
            
            
    def propagate(self,jump_radius, printFlag=False, allow_diagonals=True):
        
        #generate jumps array in reversed exponential order
        jumps_arr=[]
        jump=jump_radius
        while(jump>=1):
            jumps_arr.append(jump)
            #jump-=1
            jump//=2
            
        
        for jump in jumps_arr:
            # if(printFlag):
            #     print("jump %d"%jump)
            for offset_i in ([-jump,0,jump]):
                for offset_j in ([-jump,0,jump]):
                    if(abs(offset_i)+abs(offset_j)==0 or (abs(offset_i)+abs(offset_j)==2*jump and allow_diagonals == False)):
                        #a trivial case
                        continue
                    #nnf shifted by the jump offset
                    NNF_offset = torch.empty_like(self.NNF)
                    NNF_offset[:,:,0]=torch.clamp(self.NNF[:,:,0]-offset_i,0, self.dest_h-1)
                    NNF_offset[:,:,1]=torch.clamp(self.NNF[:,:,1]-offset_j,0, self.dest_w-1)

                    nnf_shifted=torch.roll(NNF_offset,shifts=(-offset_i,-offset_j),dims=(0,1))
                    nnd_res=self.calc_dist_parallel(self.image_src,self.image_dest,self.patch_size,nnf_shifted,self.src_h,self.src_w,self.device)
                    #Note^: the results are only correct for slice that does not contain the frame of width jump
                    #Now we will update only for the inner results
                    
                    c1 = jump if(offset_i==-1) else 0
                    c2 = nnd_res.size(0)-jump if(offset_i==1) else nnd_res.size(0)
                    c3 = jump if(offset_j==-1) else 0
                    c4 = nnd_res.size(1)-jump if(offset_j==1) else nnd_res.size(0)
                    
                    nnd_res=nnd_res[c1:c2,c3:c4]
                    
                    nnd_orig=self.NND[c1:c2,c3:c4]
                    
                    
                    cat_nnd_arrs=torch.stack([nnd_orig,nnd_res])
                    
                  
                    min_arr,argmin_arr=torch.min(cat_nnd_arrs,dim=0) #Maybe applying argmin will be faster
                    


                    cat_nnf_arrs_0=torch.stack([self.NNF[c1:c2,c3:c4,0],nnf_shifted[c1:c2,c3:c4,0]])
                    cat_nnf_arrs_1=torch.stack([self.NNF[c1:c2,c3:c4,1],nnf_shifted[c1:c2,c3:c4,1]])
                    
                    #apply argmin to the NNF
                    # permute to h x w x 0/1
                    C0=cat_nnf_arrs_0.permute(1,2,0)
                    C1=cat_nnf_arrs_1.permute(1,2,0)
                    
                    D0=C0.flatten()[2*torch.arange(C0.size(0)*C0.size(1)).to(self.device)+argmin_arr.view(-1)].view(cat_nnf_arrs_0.size(1),cat_nnf_arrs_0.size(2))
                    D1=C1.flatten()[2*torch.arange(C1.size(0)*C1.size(1)).to(self.device)+argmin_arr.view(-1)].view(cat_nnf_arrs_1.size(1),cat_nnf_arrs_1.size(2))
                    
                    D=torch.stack([D0,D1],dim=2)
                    

                    self.NND[c1:c2,c3:c4]=min_arr
                    self.NNF[c1:c2,c3:c4,:]=D
                    
                    
        
    def random_search(self,rand_search_radius, alpha, printFlag=False, alternative_prob_domain_fix=True):
        #generate radius array in reversed exponential order
        radius_arr=[]
        rad=int(rand_search_radius)
        while(rad>=1):
            radius_arr.append(int(rad))
            rad*=alpha
            
        for radius in radius_arr:
            
            #setting the NNF random offsets
            offsets=torch.randint(low=-radius,high=radius+1,size=self.NNF.size()).to(self.device)

            #probability edges fix, since many of the random cells will be outside of the frame especially around the edges, we will try to offset them towards the middle
            x_offsets=0
            y_offsets=0
            if(self.NNF.shape[0]-2*radius-2>0 and alternative_prob_domain_fix):
                #if radius is less than half of the image
                x_offsets=torch.cat([-(torch.arange(radius+1)-radius),torch.zeros(self.NNF.shape[0]-2*radius-2, dtype=int),torch.arange(radius+1)]).unsqueeze(1).to(self.device)
            if(self.NNF.shape[1]-2*radius-2>0 and alternative_prob_domain_fix):
                #if radius is less than half of the image
                y_offsets=torch.cat([-(torch.arange(radius+1)-radius),torch.zeros(self.NNF.shape[1]-2*radius-2, dtype=int),torch.arange(radius+1)]).unsqueeze(0).to(self.device)
            #   y_offsets=torch.transpose(y_offsets.unsqueeze(0),0,1)
            

            
            nnf_shifted_0=(self.NNF+offsets)[:,:,0]+x_offsets
            nnf_shifted_1=(self.NNF+offsets)[:,:,1]+y_offsets
            nnf_shifted_0=torch.clamp(nnf_shifted_0,0,nnf_shifted_0.size(0)-1)
            nnf_shifted_1=torch.clamp(nnf_shifted_1,0,nnf_shifted_1.size(1)-1)
            nnf_shifted=torch.stack([nnf_shifted_0,nnf_shifted_1],dim=2)
            
            nnd_res=self.calc_dist_parallel(self.image_src,self.image_dest,self.patch_size,nnf_shifted,self.src_h,self.src_w,self.device)
            #Note^: the results are only correct for slice that does not contain the frame of width jump
            #Now we will update only for the inner results
            
            c1 = 0
            c2 = nnd_res.size(0)
            c3 = 0
            c4 = nnd_res.size(1)
            
            nnd_res=nnd_res[c1:c2,c3:c4]
            
            nnd_orig=self.NND[c1:c2,c3:c4]                   
            
            
            cat_nnd_arrs=torch.stack([nnd_res,nnd_orig])
            
            
            min_arr,argmin_arr=torch.min(cat_nnd_arrs,dim=0) #Maybe applying argmin will be faster
            
            cat_nnf_arrs_0=torch.stack([nnf_shifted[c1:c2,c3:c4,0],self.NNF[c1:c2,c3:c4,0]])
            cat_nnf_arrs_1=torch.stack([nnf_shifted[c1:c2,c3:c4,1],self.NNF[c1:c2,c3:c4,1]])
            
            #apply argmin to the NNF
            C0=cat_nnf_arrs_0.permute(1,2,0)
            C1=cat_nnf_arrs_1.permute(1,2,0)
            
            D0=C0.flatten()[2*torch.arange(C0.size(0)*C0.size(1)).to(self.device)+argmin_arr.view(-1)].view(cat_nnf_arrs_0.size(1),cat_nnf_arrs_0.size(2))
            D1=C1.flatten()[2*torch.arange(C1.size(0)*C1.size(1)).to(self.device)+argmin_arr.view(-1)].view(cat_nnf_arrs_1.size(1),cat_nnf_arrs_1.size(2))

            D=torch.stack([D0,D1],dim=2)

            self.NND[c1:c2,c3:c4]=min_arr
            self.NNF[c1:c2,c3:c4,:]=D


    def upsample_NNF(self, factor=2):
        # factor must be an integer
        new_nnf = torch.zeros(self.NNF.size(0) * factor, self.NNF.size(1) * factor, self.NNF.size(2), dtype=torch.long).to(self.device)
        
        x=torch.arange(self.NNF.size(0)).to(self.device).repeat_interleave(factor)
        y=torch.arange(self.NNF.size(1)).to(self.device).repeat_interleave(factor)
        mesh= torch.meshgrid(x,y)
        
        
        x=torch.arange(factor).to(self.device)
        y=torch.arange(factor).to(self.device)
        
        new_nnf[:, :, 0] = self.NNF[:,:,0][mesh]*factor+x.repeat(self.NNF.size(0)).unsqueeze(1)
        new_nnf[:, :, 1] = self.NNF[:,:,1][mesh]*factor+torch.transpose(y.repeat(self.NNF.size(1)).unsqueeze(1),0,1)

        return new_nnf
        
    
    def visualize(self):
        img = torch.zeros(self.NNF.shape[0], self.NNF.shape[1], 3,dtype=torch.uint8).to(self.device)
        img[:,:,0]= (255*self.NNF[:,:,0]/ self.dest_h).int() #Red
        img[:,:,2]= (255*self.NNF[:,:,1]/ self.dest_w).int() #Blue

        return img.cpu().numpy()


    def reconstruct_avg(self, img_dst, patch_size=5):
        
        avg_window=torch.nn.AvgPool2d(patch_size, stride=1).to(self.device)
        m = torch.nn.ReflectionPad2d(patch_size // 2).to(self.device)
        img_dst = m(img_dst)[0].detach().clone().to(self.device)
        avg_dest=m(avg_window(img_dst.clone().unsqueeze(0)))[0]

        img = torch.zeros((3, self.NNF.size(0), self.NNF.size(1))).to(self.device)
        dest_i=self.NNF[:, :, 0] + self.patch_size // 2
        dest_j=self.NNF[:, :, 1] + self.patch_size // 2
        mesh=[dest_i,dest_j]
        img[0,:]= avg_dest[0][mesh]
        img[1,:]= avg_dest[1][mesh]
        img[2,:]= avg_dest[2][mesh]
        return img.permute(1,2,0).cpu().numpy()
    
    def reconstruct_without_avg(self,img_dst):
        m = torch.nn.ReflectionPad2d(self.patch_size // 2).to(self.device)
        img_dst = m(img_dst)[0].detach().clone().to(self.device)
        img = torch.zeros((3, self.NNF.size(0), self.NNF.size(1)))
        dest_i=self.NNF[:, :, 0] + self.patch_size // 2
        dest_j=self.NNF[:, :, 1] + self.patch_size // 2
        
        mesh=[dest_i,dest_j]
        img[0,:]= img_dst[0][mesh]
        img[1,:]= img_dst[1][mesh]
        img[2,:]= img_dst[2][mesh]
        return img.permute(1,2,0).cpu().numpy()
    
    def reconstruct_without_avg_with_new_nnf(self,img_dst,new_nnf):
        m = torch.nn.ReflectionPad2d(self.patch_size // 2).to(self.device)
        img_dst = m(img_dst)[0].detach().clone().to(self.device)
        img = torch.zeros((3, new_nnf.size(0), new_nnf.size(1)))
        dest_i=new_nnf[:, :, 0] + self.patch_size // 2
        dest_j=new_nnf[:, :, 1] + self.patch_size // 2
        
        mesh=[dest_i,dest_j]
        img[0,:]= img_dst[0][mesh]
        img[1,:]= img_dst[1][mesh]
        img[2,:]= img_dst[2][mesh]
        return img.permute(1,2,0).cpu().numpy()

    def get_NNF(self):
        return self.NNF

    def run(self, num_iters=10, rand_search_radius=200, alpha=0.5, jump_radius=8, allow_diagonals=True):  
        for iteration in range(1, num_iters + 1):  
            print(f"### iteration {iteration} ###")
            self.propagate(jump_radius, printFlag=True,allow_diagonals=allow_diagonals) #based on jumping flooding         
            self.random_search(rand_search_radius, alpha, printFlag=True)  
            

        return self.NNF



