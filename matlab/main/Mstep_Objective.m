function [obj, grad] = Mstep_Objective(m,prs,mu_k,var_k,muObsAll)

% reshape parameter input into variables
C = reshape(prs(1:m.dy*m.dx),[m.dy, m.dx]);
b = prs(m.dy*m.dx + 1 : end);
% get rate predictions needed for gradients for all trials
mu_h = bsxfun(@plus,mtimesx(mu_k,C'), b');
var_h = mtimesx(var_k,(C.^2)');
% neuron index to allocate spike to neuron
neuronIndex = m.neuronIndex;

if strcmp(m.nonlin.name,'Exponential')
    
    % initialise arrays for gradients
    t2 = zeros(m.ntr,1);
    grad2 = zeros(size(prs,1),m.ntr);
    
    intval = m.nonlin.Exponential(mu_h + 0.5*var_h);
    t1 = permute(m.T/m.ng*sum(reshape(intval,[m.ng*m.dy 1 m.ntr]),1),[3 2 1]);

    grad_C= m.T/m.ng*mtimesx(intval,'T',mu_k) + ...
        m.T/m.ng*squeeze(mtimesx(permute(mtimesx(permute(var_k,[1 4 2 3]),...
        permute(C,[3 1 2 4])),[5 1 2 3 4]),permute(intval,[1 5 2 4 3])));
    
    grad_b = m.T/m.ng*permute(sum(intval,1),[2 3 1]);
    grad1 = [reshape(grad_C,[],m.ntr); grad_b];
    
    for numtr = 1:m.ntr
        
        % second term containing observed values
        t2(numtr) = sum(sum(muObsAll{numtr}.*C(neuronIndex{numtr},:),2) + b(neuronIndex{numtr}));
        % gradients        
        mm1 = bsxfun(@(x,y) x == y, neuronIndex{numtr}, sparse(1:m.dy)); 

        grad_C2 = mm1'*muObsAll{numtr};        
        grad_b2 = sum(mm1,1)';
        grad2(:,numtr) = [grad_C2(:); grad_b2];
        
    end
    
else
    error('specified non-linearity not implemented')
end

% negative free energy, summed over trials
obj = sum(t1 - t2);
grad = sum(grad1 - grad2,2);

end
