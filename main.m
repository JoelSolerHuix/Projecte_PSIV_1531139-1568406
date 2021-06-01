close all
clear all

% carrega_train()       % Aquesta funció està comentada perquè amb el load de la següent línia de codi no fa falta executar-la

load('train.mat');

n = 1;      % Valor que determina quin tipus de soroll es "posarà" a les imatges

test = load('mnist_test.csv');
images_test = test(:,2:785);

r = 2;      % Valor que determina quin tipus de classificador es farà servir


if r == 1
    k = 10;
    tic;
    etiquetes_r = ModelKnnHu(images_test, images, labels, k, n); 
    toc;
    
    cont = 0;
    
    for p = 1:10
        if etiquetes_r(p) == test(p,1)
            cont = cont + 1;
        end
    end
    
    acc = (cont / 1000) * 100;
    

elseif r == 2 
   for i = 1:60000        
        temp = reshape(images(i,:), 28, 28);
        temp2 = soroll(temp, 1);
        temp_fin = treu_soroll(temp2, 1);

        arr_temp(i,:,:) = temp_fin;
   end
   
   
   k = 10;
   etiquetes = zeros(1000,1);
   tic;
   for j = 1:100
       img = images_test(j,:);
       etiqueta = ModelKnn(img, arr_temp, labels, k);
       etiqueta = etiqueta(1);
       etiquetes(j) = etiqueta;
   end
   toc;
   cont = 0;
   
   for p = 1:100
      if etiquetes(p) == test(p,1)
            cont = cont + 1;
      end
   end
   
   acc = (cont / 1000) * 100;
   
elseif r == 3  %exaustiveSearch
    for i = 1:60000        
        temp = reshape(images(i,:), 28, 28);
        temp2 = soroll(temp, 1);
        temp_fin = treu_soroll(temp2, 1);

        arr_temp(i,:,:) = temp_fin;
   end
    
    k = 10;
    Mdl = ExhaustiveSearcher(arr_temp);
    etiquetes = zeros(1000,1);
    
    tic;
    for i = 1:1000
        img = images_test(i,:);
        indexos = knnsearch(Mdl,img,'K',k);
        repetit = mode(indexos);
        etiqueta = labels(repetit);
        etiquetes(i) = etiqueta;
        
    end
    toc;
    
    cont = 0;
   
   for p = 1:1000
      if etiquetes(p) == test(p,1)
            cont = cont + 1;
      end
   end
   
   acc = (cont / 100) * 100;
    
elseif r == 4    %Kd-tree
    for i = 1:60000        
        temp = reshape(images(i,:), 28, 28);
        temp2 = soroll(temp, 1);
        temp_fin = treu_soroll(temp2, 1);

        arr_temp(i,:,:) = temp_fin;
   end
        
    tic;
    ktree = fitctree(arr_temp,labels);
    etiquetes = zeros(1000,1);
        
    for i = 1:1000
        img = images_test(i,:);
        etiqueta = predict(ktree,img);
        etiquetes(i) = etiqueta;
        
    end
    toc;
    
    cont = 0;
   
   for p = 1:1000
      if etiquetes(p) == test(p,1)
            cont = cont + 1;
      end
   end
   
   acc = (cont / 1000) * 100;

end

