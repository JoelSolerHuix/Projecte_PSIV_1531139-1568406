function etiquetes_ret = ModelKnnHu(imgs, X, labels, k, s)

hu_inv_train = zeros(60000,35);
im_veins = zeros(k);
etiquetes = zeros(k);
etiquetes_ret = zeros(10000, 1);

for i = 1:60000        
    temp = reshape(X(i,:), 28, 28);
    temp2 = soroll(temp, 1);
    temp_fin = treu_soroll(temp2, s); 
        
    hu_junt1 = parteix_i_hu(temp);
    hu_inv_train(i,:) = hu_junt1;
    
end


for i = 1:10
    distancies = zeros(60000,1);
    img = reshape(imgs(i,:), 28, 28);
    hu_junt2 = parteix_i_hu(img);
    
    for j = 1:60000
        distancia = pdist2(hu_junt2, hu_inv_train(j,:));
        distancies(j) = distancia;
    end
    
    ordenat = sort(distancies);
    for j = 1:k
        for t = 1:60000
            if ordenat(k) == distancies(t)
                im_veins(j) = t;
            end
        end
    end

    for p = 1:k
        etiquetes(p) = labels(im_veins(p));
    end

   etiquetes_ret(i) = mode(etiquetes(1));
end


end
