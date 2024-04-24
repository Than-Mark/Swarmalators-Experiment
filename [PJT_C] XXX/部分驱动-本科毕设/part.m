clc, clearvars, close all
tic


%p=parpool('local',6);
%% �����ʼֵ��
% Model
N=200; 
%��ʼ
[x0, y0, theta0,omega_n] = ic_gen(N);

Fc=4;

J=0.1;
k=1;
omega=0.5*pi;
RS=zeros();
ome=(0:0.1*omega:250*pi)';%��Դ��λ
% Computation time
t0= 0;
dt= 0.1;
tf= 200;
% ������������
partly_start = 0.1;
partly_end =0.99;
partly_step = 0.01;

% ѭ��������ͬ�� partly ֵ
for partly = partly_start:partly_step:partly_end
% �������԰ٷ�50������ʵʩ����
A = zeros(1, N); % ����ȫ0����
B=  zeros(1, N);
% ���ѡ������λ��
indices = randperm(N);
half_size = N*partly;
% ��ǰһ��λ������Ϊ1
A(indices(1:half_size)) = 1;
B(indices(1:end))=1;
%����Fij
F_p=Fc.*A;%��������
F=Fc.*B;%��ȫ����

%��������

[tv, X_p, Y_p, Theta_p]= swarma2D(x0, y0, theta0,omega_n, N, J, k, F_p, omega, t0, dt, tf);

%��ȫ����
[tv, X, Y, Theta]= swarma2D(x0, y0, theta0,omega_n, N, J, k, F, omega, t0, dt, tf);
Theta_no=Theta_p(1:10*tf,indices(half_size+1:end));
Theta_pa=Theta_p(1:10*tf,indices(1:half_size));
X_pa=X_p(:,indices(1:half_size));X_no=X_p(:,indices(half_size+1:end));
Y_pa=Y_p(:,indices(1:half_size));Y_no=Y_p(:,indices(half_size+1:end));
foldername = '1';
if ~isfolder(foldername)
    mkdir(foldername);
end % �ļ�������
filename = sprintf('�ڵ����%.2f J=%.2f K=%.2f.avi', partly, J, k);
filePath = fullfile(foldername, filename);
v = VideoWriter(filePath);
    open(v);
    an = [];
    for n = dt:0.1:tf
        subplot(1,2,1);
       snap(X, Y, Theta, dt, n)
      subplot(1,2,2);
        snap(X_pa, Y_pa, Theta_pa, dt, n)
        hold on;
        snap1(X_no, Y_no, Theta_no, dt, n)
        hold off;
       dim = [0.1 0.1 0.1 0.1];
       %str=sprintf('$F = $%.2f,$t= $%.2f',F,n); 
       str=sprintf('$node=$%.2f, $k=$%.2f, $F = $%.2f,$J = $%.2f,$t= $%.2f',partly,k,Fc,J,n);%if No floting varibale number, use %d
       delete(an)
       an = annotation('textbox',dim,'String',str,'FitBoxToText','on','Interpreter','latex');
       frame = getframe(gcf);
       drawnow;
       %frame = getframe(gcf);
       writeVideo(v, frame);
       delete(an)
    end
    close(v)
end




function [V]=orderV(X,Y,dt)
 
    V=mean(mean(sqrt(diff(X).^2+diff(Y).^2)/dt),2);  
end    
function [T]=orderT(X,Y,Theta,tf)
    phi=atan2(Y,X);
    T_plus=abs(mean(exp(1j*mean(phi(5*tf:10*tf,:)+Theta(5*tf:10*tf,:),1))));
    T_minus=abs(mean(exp(1j*mean(phi(5*tf:10*tf,:)-Theta(5*tf:10*tf,:),1))));
    T =max(T_plus,T_minus);
end    
function [Va]=orderVa(X,Y,Theta,dt)
    delta=sqrt((X-X0)^2)+(Y-Y0)^2;
    p=find(delta>5);
    delta(p)=2*pi-delta(p);
    Va=mean(mean(sqrt(diff(X).^2+diff(Y).^2+delta.^2)/dt),2);  
end
function [Vp]= orderVp(vx,vy,tf)

    phi=atan2(vy,vx);
    
    Vp=mean(abs(mean((exp(1j*phi(9*tf:10*tf,:))),2)),1);


end

function [Q]=orderphi(X,Y,tf)
  phi=atan2(Y,X);
  Q=mean(abs(mean(exp(1j*phi(9*tf:10*tf,:)),2)),1);
   

    
end

