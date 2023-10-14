clc;clear;

N_timesteps = 16;
N_spatial = 16;
alpha = 0.1;

tests = {};

ii = 1;
tests{ii}.case = 'test_case_stencil_copy_inidcator';
tests{ii}.N_spatial = N_spatial;
tests{ii}.F_init = inti_indicator(tests{ii}.N_spatial);
tests{ii}.S = [1];
tests{ii}.N_halo_x = [0 0];
tests{ii}.N_halo_y = [0 0];
tests{ii}.N_timesteps = N_timesteps;
tests{ii}.alpha = 0.1;

ii = ii+1;
tests{ii}.case = 'test_case_stencil_copy_random';
tests{ii}.N_spatial = 64;
tests{ii}.F_init = inti_random(tests{ii}.N_spatial);
tests{ii}.S = [1];
tests{ii}.N_halo_x = [0 0];
tests{ii}.N_halo_y = [0 0];
tests{ii}.N_timesteps = N_timesteps;
tests{ii}.alpha = 0.1;

ii = ii+1;
tests{ii}.case = 'test_case_stencil_1D_i1_inidcator';
tests{ii}.N_spatial = N_spatial;
tests{ii}.F_init = inti_indicator(tests{ii}.N_spatial);
tests{ii}.S = [1;1];
tests{ii}.N_halo_x = [0 0];
tests{ii}.N_halo_y = [1 0];
tests{ii}.N_timesteps = N_timesteps;
tests{ii}.alpha = 0.1;

ii = ii+1;
tests{ii}.case = 'test_case_stencil_1D_i1_random';
tests{ii}.N_spatial = N_spatial;
tests{ii}.F_init = inti_random(tests{ii}.N_spatial);
tests{ii}.S = [1;1];
tests{ii}.N_halo_x = [0 0];
tests{ii}.N_halo_y = [1 0];
tests{ii}.N_timesteps = N_timesteps;
tests{ii}.alpha = 0.1;

ii = ii+1;
tests{ii}.case = 'test_case_stencil_1D_i2_inidcator';
tests{ii}.N_spatial = N_spatial;
tests{ii}.F_init = inti_indicator(tests{ii}.N_spatial);
tests{ii}.S = [1;1;1];
tests{ii}.N_halo_x = [0 0];
tests{ii}.N_halo_y = [1 1];
tests{ii}.N_timesteps = N_timesteps;
tests{ii}.alpha = 0.1;

ii = ii+1;
tests{ii}.case = 'test_case_stencil_1D_i2_random';
tests{ii}.N_spatial = N_spatial;
tests{ii}.F_init = inti_random(tests{ii}.N_spatial);
tests{ii}.S = [1;1;1];
tests{ii}.N_halo_x = [0 0];
tests{ii}.N_halo_y = [1 1];
tests{ii}.N_timesteps = N_timesteps;
tests{ii}.alpha = 0.1;

ii = ii+1;
tests{ii}.case = 'test_case_stencil_1D_j1_inidcator';
tests{ii}.N_spatial = N_spatial;
tests{ii}.F_init = inti_indicator(tests{ii}.N_spatial);
tests{ii}.S = [1 1];
tests{ii}.N_halo_x = [1 0];
tests{ii}.N_halo_y = [0 0];
tests{ii}.N_timesteps = N_timesteps;
tests{ii}.alpha = 0.1;

ii = ii+1;
tests{ii}.case = 'test_case_stencil_1D_j1_random';
tests{ii}.N_spatial = N_spatial;
tests{ii}.F_init = inti_random(tests{ii}.N_spatial);
tests{ii}.S = [1 1];
tests{ii}.N_halo_x = [1 0];
tests{ii}.N_halo_y = [0 0];
tests{ii}.N_timesteps = N_timesteps;
tests{ii}.alpha = 0.1;

ii = ii+1;
tests{ii}.case = 'test_case_stencil_1D_j2_inidcator';
tests{ii}.N_spatial = N_spatial;
tests{ii}.F_init = inti_indicator(tests{ii}.N_spatial);
tests{ii}.S = [1 1 1];
tests{ii}.N_halo_x = [1 1];
tests{ii}.N_halo_y = [0 0];
tests{ii}.N_timesteps = N_timesteps;
tests{ii}.alpha = 0.1;

ii = ii+1;
tests{ii}.case = 'test_case_stencil_1D_j2_random';
tests{ii}.N_spatial = N_spatial;
tests{ii}.F_init = inti_random(tests{ii}.N_spatial);
tests{ii}.S = [1 1 1];
tests{ii}.N_halo_x = [1 1];
tests{ii}.N_halo_y = [0 0];
tests{ii}.N_timesteps = N_timesteps;
tests{ii}.alpha = 0.1;

ii = ii+1;
tests{ii}.case = 'test_case_stencil_2D_inidcator';
tests{ii}.N_spatial = N_spatial;
tests{ii}.F_init = inti_indicator(tests{ii}.N_spatial);
tests{ii}.S = [0 1 0;
               1 1 1;
               0 1 0];
tests{ii}.N_halo_x = [1 1];
tests{ii}.N_halo_y = [1 1];
tests{ii}.N_timesteps = N_timesteps;
tests{ii}.alpha = 0.1;

ii = ii+1;
tests{ii}.case = 'test_case_stencil_2D_random';
tests{ii}.N_spatial = N_spatial;
tests{ii}.F_init = inti_random(tests{ii}.N_spatial);
tests{ii}.S = [0 1 0;
               1 1 1;
               0 1 0];
tests{ii}.N_halo_x = [1 1];
tests{ii}.N_halo_y = [1 1];
tests{ii}.N_timesteps = N_timesteps;
tests{ii}.alpha = 0.1;

for jj=1:length(tests)
    
    write_matrix(zero_halo(tests{jj}.F_init,tests{jj}.N_halo_x,tests{jj}.N_halo_y),[tests{jj}.case,'_init.csv']);
    write_matrix(tests{jj}.S,[tests{jj}.case,'_stencil.csv']);

    F_out = apply_stencil(tests{jj}.F_init,tests{jj}.S,tests{jj}.N_halo_x,tests{jj}.N_halo_y);
    write_matrix(zero_halo(F_out,tests{jj}.N_halo_x,tests{jj}.N_halo_y),[tests{jj}.case,'_out.csv']);

    Fp1 = tests{jj}.F_init;
    for i=1:tests{jj}.N_timesteps
        F = Fp1;
        Fs = apply_stencil(F,tests{jj}.S,tests{jj}.N_halo_x,tests{jj}.N_halo_y);
        Fp1 = time_step(F,Fs,tests{jj}.alpha);
    end

    write_matrix(zero_halo(Fp1,tests{jj}.N_halo_x,tests{jj}.N_halo_y),[tests{jj}.case,'_timesteps_',num2str(tests{jj}.N_timesteps),'_alpha_',num2str(tests{jj}.alpha),'.csv'])

end

function [] = write_matrix(M,file_name)
    writelines(strcat(num2str(size(M,1)),",",num2str(size(M,2))),file_name);
    writematrix(M,file_name,'WriteMode','append')
end

function H = inti_random(N_spatial)
    H = rand(N_spatial,N_spatial);
end

function H = inti_indicator(N_spatial)
    x = linspace(-2,2,N_spatial);
    x_b = abs(x) < 1;
    H = x_b'*x_b;
end

function Hnp1 = time_step(H,Hs,alpha)
    Hnp1 = H - alpha*Hs;
end

function Hs = apply_stencil(H,S,N_halo_x,N_halo_y)
    Hs = conv2(zero_halo(H,N_halo_x,N_halo_y),S,'valid');
end

function Hhv = periodic_halo(H,N_halo_x,N_halo_y)
    Hh = [H(:,end-N_halo_x(1)+1:end) H H(:,1:N_halo_x(2))];
    Hhv = [Hh(end-N_halo_y(1)+1:end,:); Hh; Hh(1:N_halo_y(2),:)];
end

function Hhv = zero_halo(H,N_halo_x,N_halo_y)
    m = size(H,1);
    Hh = [zeros(m,N_halo_x(1)) H zeros(m,N_halo_x(2))];
    n = size(Hh,2);
    Hhv = [zeros(N_halo_y(1),n); Hh; zeros(N_halo_y(2),n)];
end