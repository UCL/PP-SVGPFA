function m = initialise_VariationalPrs_svGPFA(m);

for ii = 1:m.dx
    m.q_mu{ii} = zeros(m.numZ(ii),1,m.ntr);
    m.q_sqrt{ii} = repmat(vec(0.01*eye(m.numZ(ii),m.opts.varRnk(ii))),[1 1 m.ntr]);
    m.q_diag{ii} = repmat(0.01*ones(m.numZ(ii),1),[1 1 m.ntr]);
    
    qq =  reshape(m.q_sqrt{ii},m.numZ(ii),m.opts.varRnk(ii),m.ntr);
    dd = diag3D(m.q_diag{ii}.^2);
    m.q_sigma{ii} = mtimesx(qq,qq,'T') + dd;
end
