function [x0, y0, theta0,omega_n] = ic_gen(N)
% This function allows to generate uniform initial conditions for the
% swarmalators model between a specified range.
% Inputs:
%         N : Number of particles;
%
% Outputs:
%         x0:
%         y0:
%         theta0:
%

pmax = 1;                       pmin = -1;
thmax = pi;                     thmin = -pi;


x0 = (pmax-pmin).*rand(N,1) + pmin;
y0 = (pmax-pmin).*rand(N,1) + pmin;
theta0 = (thmax-thmin).*rand(N,1) + thmin;
% ���ɾ��ȷֲ�����ȻƵ��
omega_n = linspace(thmin, thmax, N);
% ���������ȻƵ��˳��
omega_n = omega_n(randperm(N));%�þ���ΪN�����ӵ���ȻƵ�ʾ���
end