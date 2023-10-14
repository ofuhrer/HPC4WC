clc;clear;

N_timesteps = 100;
N_spatial = 16;
alpha = 0.1;

ii = 1;
tests{ii}.case = 'test_case_stencil_3D_inidcator';
tests{ii}.F_init = inti_indicator(N_spatial);
tests{ii}.S(:,:,1) = [0 0 0;
                      0 1 0;
                      0 0 0];
tests{ii}.S(:,:,2) = [0 1 0;
                      1 1 1;
                      0 1 0];
tests{ii}.S(:,:,3) = [0 0 0;
                      0 1 0;
                      0 0 0];
tests{ii}.N_halo_x = [1 1];
tests{ii}.N_halo_y = [1 1];
tests{ii}.N_halo_z = [1 1];
tests{ii}.N_timesteps = N_timesteps;
tests{ii}.N_spatial = N_spatial;
tests{ii}.alpha = alpha;

ii = ii+1;
tests{ii}.case = 'test_case_stencil_3D_random';
tests{ii}.F_init = inti_random(N_spatial);
tests{ii}.S(:,:,1) = [0 0 0;
                      0 1 0;
                      0 0 0];
tests{ii}.S(:,:,2) = [0 1 0;
                      1 1 1;
                      0 1 0];
tests{ii}.S(:,:,3) = [0 0 0;
                      0 1 0;
                      0 0 0];
tests{ii}.N_halo_x = [1 1];
tests{ii}.N_halo_y = [1 1];
tests{ii}.N_halo_z = [1 1];
tests{ii}.N_timesteps = N_timesteps;
tests{ii}.N_spatial = N_spatial;
tests{ii}.alpha = alpha;


for jj=1:length(tests)

    write_matrix(tests{jj}.S,[tests{jj}.case,'_stencil.csv'])
    write_matrix(zero_halo(tests{jj}.F_init,tests{jj}.N_halo_x,tests{jj}.N_halo_y,tests{jj}.N_halo_z),[tests{jj}.case,'_init.csv'])
    
    F_out = apply_stencil(tests{jj}.F_init,tests{jj}.S,tests{jj}.N_halo_x,tests{jj}.N_halo_y,tests{jj}.N_halo_z);
    write_matrix(zero_halo(F_out,tests{jj}.N_halo_x,tests{jj}.N_halo_y,tests{jj}.N_halo_z),[tests{jj}.case,'_out.csv'])
    
    Fp1 = tests{jj}.F_init;
    for i=1:tests{jj}.N_timesteps
        F = Fp1;
        Fs = apply_stencil(F,tests{jj}.S,tests{jj}.N_halo_x,tests{jj}.N_halo_y,tests{jj}.N_halo_z);
        Fp1 = time_step(F,Fs,tests{jj}.alpha);
    end
    
    write_matrix(zero_halo(Fp1,tests{jj}.N_halo_x,tests{jj}.N_halo_y,tests{jj}.N_halo_z),[tests{jj}.case,'_timesteps_',num2str(tests{jj}.N_timesteps),'_alpha_',num2str(tests{jj}.alpha),'.csv']);

end


function [] = write_matrix(M,file_name)
    writelines(strcat(num2str(size(M,1)),",",num2str(size(M,2)),",",num2str(size(M,3))),file_name);
    writematrix(M,file_name,'WriteMode','append');
end

function Hs = apply_stencil(H,S,N_halo_x,N_halo_y,N_halo_z)
    Hs = convn(zero_halo(H,N_halo_x,N_halo_y,N_halo_z),S,'valid');
end

function Hh = zero_halo(H,N_halo_x,N_halo_y,N_halo_z)
    m = size(H,1);
    n = size(H,2);
    l = size(H,3);
    Hh = zeros(m+sum(N_halo_x),n+sum(N_halo_y),l+sum(N_halo_z));
    Hh(1+N_halo_x(1):end-N_halo_x(2),1+N_halo_y(1):end-N_halo_y(2),1+N_halo_z(1):end-N_halo_z(2)) = H;
end

function H3 = inti_indicator(N_spatial)
    x = linspace(-2,2,N_spatial);
    x_b = abs(x) < 1;
    H = x_b'*x_b;

    H3 = zeros(N_spatial,N_spatial,N_spatial);

    for i=1:length(x_b)
        H3(:,:,i) = x_b(i)*H;
    end
end

function Hnp1 = time_step(H,Hs,alpha)
    Hnp1 = H - alpha*Hs;
end

function H3 = inti_random(N_spatial)
    H3 = rand(N_spatial,N_spatial,N_spatial);
end