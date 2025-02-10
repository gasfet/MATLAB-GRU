function gru02()
    % 함수는 GRU(Gated Recurrent Unit) 모델을 사용하여 텍스트 데이터를 학습하고, 학습된 모델을 통해 텍스트를 생성하는 역할을 합니다. 이 함수는 다음과 같은 주요 작업을 수행합니다:
    % 주요 작업
    % 1.    데이터 읽기 및 전처리
    % •    data = read_raw('alice29.txt');: 텍스트 파일을 읽어와서 원시 바이트 스트림으로 저장합니다.
    % •    text_length = size(data, 1);: 텍스트 데이터의 길이를 계산합니다.
    % •    symbols = unique(data);: 텍스트 데이터에서 고유한 문자를 추출합니다.
    % •    alphabet_size = size(symbols, 1);: 고유한 문자의 수를 계산합니다.
    % •    codes = eye(n_in);: 원-핫 인코딩을 위한 코드 매트릭스를 생성합니다.
    % 2.    하이퍼파라미터 및 모델 파라미터 초기화
    % •    hidden_size, seq_length, learning_rate, vocab_size 등의 하이퍼파라미터를 설정합니다.
    % •    GRU 모델의 가중치와 바이어스를 초기화합니다 (Wz, Uz, bz, Wr, Ur, br, Wu, Uu, bu, Why, by).
    % 3.    학습 루프
    % •    for e = 1:max_epochs: 최대 에포크 수만큼 학습을 반복합니다.
    % •    sub1 함수를 호출하여 각 에포크에서 텍스트 데이터를 여러 시퀀스로 나누어 처리합니다.
    % 4.    순전파 및 역전파
    % •    sub2 함수에서 순전파(sub9)와 역전파(sub8)를 수행하여 모델의 가중치를 업데이트합니다.
    % •    probs 변수를 사용하여 각 시간 단계에서 모델이 예측한 다음 문자의 확률 분포를 계산합니다.
    % •    교차 엔트로피 손실을 계산하고, 이를 통해 모델의 성능을 평가합니다.
    % 5.    기울기 클리핑 및 가중치 업데이트
    % •    sub7 함수에서 기울기를 클리핑하여 기울기 폭발을 방지합니다.
    % •    sub6 함수에서 Adagrad 알고리즘을 사용하여 가중치와 바이어스를 업데이트합니다.
    % 6.    상태 업데이트 및 통계 출력
    % •    sub5 함수에서 각 상태(z, r, u, rh, h, y, probs)를 다음 시간 단계로 이동합니다.
    % •    sub4 함수에서 손실 기록을 업데이트하고, 일정 시간마다 학습 통계를 출력하며, 학습된 모델을 사용하여 텍스트를 생성합니다.
    %read raw byte stream
    data = read_raw('alice29.txt');

    text_length = size(data, 1);

    % alphabet
    symbols = unique(data);
    alphabet_size = size(symbols, 1);
    ASCII_SIZE = 256;

    % n_in - size of the alphabet, ex. 4 (ACGT)
    n_in = ASCII_SIZE;
    % n_out = n_in - predictions have the same size as inputs
    n_out = n_in;

    codes = eye(n_in);  % 원-hot인코딩에서 codes는 하나의 알파벳이다. 

    max_iterations = text_length;
    max_epochs = 1000;

    observations = zeros(1, text_length);
    perc = round(text_length / 100);
    show_every = 10; %show stats every show_every s

    % hyperparameters
    hidden_size = 256; % size of hidden layer of neurons
    seq_length = 50; % number of steps to unroll the RNN for  ==> 정답을 만들기 위해 필요한 시점 개수 time step
    learning_rate = 1e-1;
    vocab_size = n_in;

    % model parameters
    Wz = randn(hidden_size, vocab_size) * 0.01; % x to update gate
    Uz = randn(hidden_size, hidden_size) * 0.01; % h_prev to update gate
    bz = zeros(hidden_size, 1); %input gate bias

    Wr = randn(hidden_size, vocab_size) * 0.01; % x to reset gate
    Ur = randn(hidden_size, hidden_size) * 0.01; % h_prev to reset gate
    br = zeros(hidden_size, 1); %output gate bias

    Wu = randn(hidden_size, vocab_size) * 0.01; % x to candidate gate
    Uu = randn(hidden_size, hidden_size) * 0.01; % h_prev to candidate gate
    bu = zeros(hidden_size, 1); %input gate bias

    Why = randn(vocab_size, hidden_size) * 0.01; % hidden to output => 히든 상태에서 출력으로 가는 가중치 행렬
    by = zeros(vocab_size, 1); % output bias

    h = zeros(hidden_size, seq_length); % hidden
    u = zeros(hidden_size, seq_length); % candidates
    z = zeros(hidden_size, seq_length); % update gates
    r = zeros(hidden_size, seq_length); % reset gates
    rh = zeros(hidden_size, seq_length); % r(t) .* h(t-1)

    %adagrad memory
    mWhy = zeros(size(Why));
    mby = zeros(size(by));

    mWz = zeros(size(Wz));
    mUz = zeros(size(Uz));
    mbz = zeros(size(bz));

    mWr = zeros(size(Wr));
    mUr = zeros(size(Ur));
    mbr = zeros(size(br));

    mWu = zeros(size(Wu));
    mUu = zeros(size(Uu));
    mbu = zeros(size(bu));

    % •	의미: target 변수는 모델이 예측해야 하는 실제 타겟 값을 저장합니다. 이는 입력 시퀀스의 다음 문자를 원-핫 인코딩 형태로 나타낸 것입니다.
    % •	크기: (vocab_size, seq_length)
    % •	역할: 각 시간 단계에서 모델이 예측해야 하는 실제 문자의 원-핫 인코딩 벡터를 저장합니다. 예를 들어, 입력 시퀀스가 "hello"라면, target은 "elloh"의 원-핫 인코딩 벡터가 됩니다.
    target = zeros(vocab_size, seq_length);

    % •	의미: y 변수는 모델의 출력 값을 저장합니다. 이는 GRU 레이어의 출력 히든 상태를 출력 레이어를 통해 변환한 값입니다.
    % •	크기: (vocab_size, seq_length)
    % •	역할: 각 시간 단계에서 모델이 예측한 다음 문자의 로짓(logit) 값을 저장합니다. 이 값은 소프트맥스 함수에 입력되어 확률 분포로 변환됩니다.
    y = zeros(vocab_size, seq_length);

    % •	의미: dy 변수는 출력의 기울기(gradient)를 저장합니다. 이는 역전파 과정에서 사용됩니다.
    % •	크기: (vocab_size, seq_length)
    % •	역할: 각 시간 단계에서 모델의 출력과 실제 타겟 값 간의 차이를 저장합니다. 이는 역전파 과정에서 기울기를 계산하는 데 사용됩니다.
    dy = zeros(vocab_size, seq_length);
    % 변수 probs는 GRU 모델의 출력 확률을 저장하는 역할을 합니다. 각 시간 단계에서 모델이 예측한 다음 문자의 확률 분포를 나타냅니다.
    % 역할
    % 1.    출력 확률 계산: probs는 각 시간 단계에서 모델이 예측한 다음 문자의 확률 분포를 저장합니다. 이는 소프트맥스 함수에 의해 계산됩니다.
    % 2.    손실 계산: probs는 교차 엔트로피 손실을 계산하는 데 사용됩니다. 모델의 출력 확률과 실제 타겟 확률 간의 차이를 기반으로 손실을 계산합니다.
    % 3.    역전파: probs는 역전파 과정에서 기울기를 계산하는 데 사용됩니다. 출력 확률과 실제 타겟 확률 간의 차이를 기반으로 기울기를 계산합니다.
    probs = zeros(vocab_size, seq_length);

    %using log2 (bits), initial guess
    smooth_loss = - log2(1.0 / alphabet_size);
    loss_history = [];

    %reset timer
    tic

    for e = 1:max_epochs
        sub1();
    end

    function sub1()

        % sub1 함수에서 sub2 함수를 반복 호출하는 이유는 GRU 모델을 학습하기 위해 텍스트 데이터를 여러 시퀀스로 나누어 처리하기 위함입니다. 각 시퀀스는 RNN의 한 번의 학습 단계에 해당하며, 이 과정을 통해 모델이 텍스트 데이터를 학습하게 됩니다.
        % 반복 호출의 이유
        % •    시퀀스 학습: GRU 모델은 시퀀스 데이터를 학습하는데 적합합니다. 텍스트 데이터를 일정 길이의 시퀀스로 나누어 각 시퀀스를 순차적으로 학습합니다.
        % •    메모리 효율성: 전체 텍스트 데이터를 한 번에 처리하는 대신, 작은 시퀀스로 나누어 처리함으로써 메모리 사용을 효율적으로 관리할 수 있습니다.
        % •    모델 업데이트: 각 시퀀스에 대해 순전파와 역전파를 수행하여 모델의 가중치를 업데이트합니다. 이를 통해 모델이 점진적으로 학습됩니다.

        %set some random context
        h(:, 1) = tanh(randn(size(h(:, 1))));

        %or zeros
        %h(:, 1) = zeros(size(h(:, 1)));

        % •    beginning: 시작 지점을 랜덤하게 설정합니다. 이는 모델이 텍스트의 다양한 부분을 학습할 수 있도록 하기 위함입니다.
        % •    ii: 반복 변수로, beginning부터 seq_length 간격으로 증가합니다.
        % •    max_iterations - seq_length: 반복의 종료 지점으로, 텍스트 데이터의 끝에서 seq_length를 뺀 값입니다. 이는 마지막 시퀀스가 텍스트 범위를 벗어나지 않도록 하기 위함입니다.

        beginning = randi([2 1+seq_length]); %randomize starting point
        for ii = beginning:seq_length:max_iterations - seq_length
            sub2();
        end



        function sub2()

            % 1.    기울기 초기화: dby, dWhy, dy, dr, dz, du, drh, dUz, dUr, dUu, dWz, dWr, dWu, dbz, dbr, dbu, dhnext를 0으로 초기화합니다.
            % 2.    다음 심볼을 가져옵니다: xs와 target을 설정합니다.
            % 3.    순전파 수행: sub9 함수를 호출하여 순전파를 수행하고 손실을 계산합니다.
            % 4.    역전파 수행: sub8 함수를 호출하여 역전파를 수행하고 기울기를 계산합니다.
            % 5.    기울기 클리핑: sub7 함수를 호출하여 기울기를 클리핑합니다.
            % 6.    가중치 조정: sub6 함수를 호출하여 가중치를 조정합니다.
            % 7.    상태 업데이트: sub5 함수를 호출하여 상태를 업데이트합니다.
            % 8.    손실을 부드럽게 업데이트하고, 일정 시간마다 통계를 출력하고 텍스트를 생성합니다: sub4 함수를 호출합니다.

            % reset grads
            dby = zeros(size(by));
            dWhy = zeros(size(Why));
            dy = zeros(size(target));
            dr = zeros(size(r));
            dz = zeros(size(z));
            du = zeros(size(u));
            drh = zeros(size(rh));

            dUz = zeros(size(Uz));
            dUr = zeros(size(Ur));
            dUu = zeros(size(Uu));

            dWz = zeros(size(Wz));
            dWr = zeros(size(Wr));
            dWu = zeros(size(Wu));

            dbz = zeros(size(bz));
            dbr = zeros(size(br));
            dbu = zeros(size(bu));

            dhnext = zeros(size(h(:, 1)));

            % get next symbol
            % xs 행렬은 256개의 행에서 열이 1부터 50까지 있으며, data에 대항하는 문자가 있는 곳에 값을 1을
            % 가진다.
            xs(:, 1:seq_length) = codes(data(ii - 1:ii + seq_length - 2), :)';
            target(:, 1:seq_length) = codes(data(ii:ii + seq_length - 1), :)';

            observations = char(data(ii - 1:ii + seq_length - 2))';
            t_observations = char(data(ii:ii + seq_length - 1))';

            % forward pass:

            loss = 0;
            sub10();

            %bits/symbol
            loss = loss/seq_length;

            % backward pass:   역전파 및 기울기 계산
            sub11();

            elapsed = toc;

            % debug code, checks gradients - slow!
            % if (elapsed > show_every)
            %     gru_grad_check;
            % end

            sub12();
            sub13();

            % %%%%%%%%%%%%%%%%%%%%%

            smooth_loss = smooth_loss * 0.999 + loss * 0.001;

            % show stats every show_every s
            if (elapsed > show_every)
                sub14();
                drawnow;
                % reset timer
                tic
            end
            sub15();

            function sub10()
                % sub9 함수는 순전파를 수행합니다:
                % 1.    각 시간 단계 t에 대해 GRU 게이트를 계산합니다: z (업데이트 게이트), r (리셋 게이트), rh (리셋된 히든 상태), u (후보 히든 상태), h (새 히든 상태).
                % 2.    출력 y를 계산하고 확률 probs를 계산합니다.
                % 3.    교차 엔트로피 손실을 계산합니다.
                for t = 2:seq_length

                    % GRU gates
                    % TODO: z and r can be computed at the same time with a single MMU
                    z(:, t) = sigmoid(Wz * xs(:, t) + Uz * h(:, t - 1) + bz); % update gates
                    r(:, t) = sigmoid(Wr * xs(:, t) + Ur * h(:, t - 1) + br); % reset gates

                    % r(t) .* h(t-1)
                    rh(:, t) = r(:, t) .* h(:, t - 1);

                    % candidate value for hidden state : 여기서 u는 일반적으로  h^tild표현이다. 
                    u(:, t) = tanh(Wu * xs(:, t) + Uu * rh(:, t) + bu);

                    % new hidden state  :  최종 hidden state
                    h(:, t) = (1.0 - z(:, t)) .* h(:, t - 1) + z(:, t) .* u(:, t);

                    % update y 출력 계산
                    y(:, t) = Why * h(:, t) + by;

                    % compute probs
                    probs(:, t) = exp(y(:, t)) ./ sum(exp(y(:, t)));

                    % cross-entropy loss, sum logs of probabilities of target outputs
                    loss = loss + sum( -log2(probs(:, t)) .* target(:, t));

                end
            end

            function sub11()
                % sub8 함수는 역전파를 수행합니다:
                % 1.    각 시간 단계 t에 대해 출력의 기울기 dy를 계산합니다.
                % 2.    GRU 게이트의 기울기 dz, du, drh, dr를 계산합니다.
                % 3.    선형 계층의 기울기 dUz, dWz, dbz, dUr, dWr, dbr, dUu, dWu, dbu를 계산합니다.
                % 4.    다음 히든 상태의 기울기 dhnext를 계산합니다.
                % backward pass:   역전파 및 기울기 계산
                for t = seq_length: - 1:2

                    % dy (global error)
                    dy(:, t) = probs(:, t) - target(:, t); %dy[targets[t]] -= 1 # backprop into y
                    dWhy = dWhy + dy(:, t) * h(:, t)'; %dWhy += np.doutt(dy, hs[t].T)
                    dby = dby + dy(:, t); % dby += dy
                    dh = Why' * dy(:, t) + dhnext; %dh = np.dot(Why.T, dy) + dhnext

                    %dz = dh * (u(t) - h(t-1)) * sigmoid'(z(t))
                    dz = dh .* (u(:, t) - h(:, t - 1)) .* (z(:, t) .* (1.0 - z(:, t)));
                    %du = dh * (1 - z(t)) * tanh'(u(t))
                    du = dh .* (1.0 - u(:, t) .* u(:, t)) .* z(:, t);
                    %drh = Uu' * du
                    drh = Uu' * du;
                    %dr = drh * h(t-1) * sigmoid'(r(t))
                    dr = drh .* h(:, t - 1) .* (r(:, t) .* (1.0 - r(:, t)));

                    %linear layers
                    dUz = dUz + dz * h(:, t - 1)';
                    dWz = dWz + dz * xs(:, t)';
                    dbz = dbz + dz;

                    dUr = dUr + dr * h(:, t - 1)';
                    dWr = dWr + dr * xs(:, t)';
                    dbr = dbr + dr;

                    dUu = dUu + du * rh(:, t)';
                    dWu = dWu + du * xs(:, t)';
                    dbu = dbu + du;

                    dhnext = Ur' * dr + Uz' * dz + drh .* r(:, t) + dh .* (1.0 - z(:, t));

                end
            end

            function sub12()
                % sub7 함수는 기울기를 클리핑합니다:
                % 1.    각 기울기 dWhy, dby, dUz, dWz, dbz, dUr, dWr, dbr, dUu, dWu, dbu를 지정된 범위로 클리핑합니다.
                % clip gradients to some range
                dWhy = clip(dWhy, -5, 5);
                dby = clip(dby, -5, 5);

                dUz = clip(dUz, -5, 5);
                dWz = clip(dWz, -5, 5);
                dbz = clip(dbz, -5, 5);

                dUr = clip(dUr, -5, 5);
                dWr = clip(dWr, -5, 5);
                dbr = clip(dbr, -5, 5);

                dUu = clip(dUu, -5, 5);
                dWu = clip(dWu, -5, 5);
                dbu = clip(dbu, -5, 5);
            end

            function sub13()
                % sub6 함수는 가중치를 조정합니다:
                % 1.    Adagrad를 사용하여 가중치와 바이어스를 업데이트합니다: Why, by, Uz, Wz, bz, Ur, Wr, br, Uu, Wu, bu.
                % % adjust weights, adagrad:  가중치 계산
                mWhy = mWhy + dWhy .* dWhy;
                mby = mby + dby .* dby;

                mUz = mUz + dUz .* dUz;
                mWz = mWz + dWz .* dWz;
                mbz = mbz + dbz .* dbz;

                mUr = mUr + dUr .* dUr;
                mWr = mWr + dWr .* dWr;
                mbr = mbr + dbr .* dbr;

                mUu = mUu + dUu .* dUu;
                mWu = mWu + dWu .* dWu;
                mbu = mbu + dbu .* dbu;

                Why = Why - learning_rate * dWhy ./ (sqrt(mWhy + eps));
                by = by - learning_rate * dby ./ (sqrt(mby + eps));

                Uz = Uz - learning_rate * dUz ./ (sqrt(mUz + eps));
                Wz = Wz - learning_rate * dWz ./ (sqrt(mWz + eps));
                bz = bz - learning_rate * dbz ./ (sqrt(mbz + eps));

                Ur = Ur - learning_rate * dUr ./ (sqrt(mUr + eps));
                Wr = Wr - learning_rate * dWr ./ (sqrt(mWr + eps));
                br = br - learning_rate * dbr ./ (sqrt(mbr + eps));

                Uu = Uu - learning_rate * dUu ./ (sqrt(mUu + eps));
                Wu = Wu - learning_rate * dWu ./ (sqrt(mWu + eps));
                bu = bu - learning_rate * dbu ./ (sqrt(mbu + eps));
            end

            function sub14()
                % sub4 함수는 통계를 출력하고 텍스트를 생성합니다:
                % 1.    손실 기록을 업데이트합니다.
                % 2.    현재 에포크와 손실을 출력합니다.
                % 3.    학습된 모델을 사용하여 텍스트를 생성합니다.
                % 4.    플롯을 업데이트하여 학습 과정을 시각화합니다.
                loss_history = [loss_history smooth_loss];

                fprintf('[epoch %d] %d %% text read... smooth loss = %.3f\n', e, round(100 * ii / text_length), smooth_loss);
                fprintf('\n\nGenerating some text...\n');

                % random h,c seeds
                % 학습된 모델로 Text를 생성한다.
                t = generate_gru(    Uz, Wz, bz, ...
                    Ur, Wr, br, ...
                    Uu, Wu, bu, ...
                    Why, by, ...
                    500, tanh(randn(size(h(:, 1)))));
                %t = generate_rnn(Wxh, Whh, Why, bh, by, 1000, clip(randn(size(Why, 2), 1) * 0.5, -1, 1));
                %generate according to the last seen h
                % t = generate_gru(    Uz, Wz, bz, ...
                %     Ur, Wr, br, ...
                %     Uu, Wu, bu, ...
                %     Why, by, ...
                %     500, h(:, seq_length));

                fprintf('%s \n', t);

                % update plots
                figure(1)

                subplot(6, 6, 1);
                imagesc(z + 0.5);
                title('z gates');
                subplot(6, 6, 7);
                imagesc(Wz');
                title('Wz');
                subplot(6, 6, 13);
                imagesc(Uz);
                title('Uz');

                subplot(6, 6, 2);
                imagesc(r + 0.5);
                title('r gates');
                subplot(6, 6, 8);
                imagesc(Wr');
                title('Wr');

                subplot(6, 6, 19);
                imagesc(dUz);
                title('dUz');

                subplot(6, 6, 14);
                imagesc(Ur);
                title('Ur');

                subplot(6, 6, 20);
                imagesc(dUr);
                title('dUr');

                subplot(6, 6, 21);
                imagesc(dUu);
                title('dUu');

                subplot(6, 6, 25);
                imagesc(dWz');
                title('dWz');

                subplot(6, 6, 26);
                imagesc(dWr');
                title('dWr');

                subplot(6, 6, 27);
                imagesc(dWu');
                title('dWu');

                subplot(6, 6, 3);
                imagesc(rh + 0.5);
                title('rh');

                subplot(6, 6, 4);
                imagesc(u + 0.5);
                title('u');
                subplot(6, 6, 10);
                imagesc(Wu');
                title('Wu');
                subplot(6, 6, 16);
                imagesc(Uu);
                title('Uu');

                subplot(6, 6, 6);
                imagesc((h + 1) / 2);
                title('h');

                subplot(6, 6, 11);
                imagesc(Why);
                title('Why');

                subplot(6, 6, 12);
                imagesc(dWhy);
                title('dWhy');

                subplot(6, 6, 18);
                plot(loss_history);
                title('Loss history');

                subplot(6, 6, 31);
                imagesc(xs);
                str = sprintf('xs: [%s]', observations);
                title(str);

                subplot(6, 6, 32);
                imagesc(target);
                str = sprintf('targets: [%s]', t_observations);
                title(str);

                subplot(6, 6, 33);
                imagesc(probs);
                title('probs');

                subplot(6, 6, 34);
                imagesc(dy);
                title('dy');
            end

            function sub15()
                % sub5 함수는 상태를 업데이트합니다:
                % 1.    각 상태 z, r, u, rh, h, y, probs를 다음 시간 단계로 이동합니다.
                %carry
                z(:, 1) = z(:, seq_length);
                r(:, 1) = r(:, seq_length);
                u(:, 1) = u(:, seq_length);
                rh(:, 1) = rh(:, seq_length);
                h(:, 1) = h(:, seq_length);
                y(:, 1) = y(:, seq_length);
                probs(:, 1) = probs(:, seq_length);
            end


        end

    end


end
