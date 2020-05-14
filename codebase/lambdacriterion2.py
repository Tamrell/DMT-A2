

import torch
torch.set_printoptions(threshold=10000)


def train_loop_plug(net, dataset, optimizer, sigma, device):
    net[0].train()
    net[1].train()

    net[0].zero_grad()
    net[1].zero_grad()

    # print("FORCE TESTING lr: 1e-3")

    optimizers = []
    optimizers.append(torch.optim.Adam(net[0].parameters(), lr=1e-3))
    optimizers.append(torch.optim.Adam(net[1].parameters(), lr=1e-3))

    count = 0
    batch_size = 200
    grad_batch, y_pred_batch = [], []
    batch_counter = 0
    for search_id, X, Y, rand_bool, props in dataset:
        if torch.sum(Y) == 0:
            # negative session, cannot learn useful signal
            continue
        X = X.to(device)
        Y = Y.to(device)
        props=props.to(device)

        with torch.no_grad():
            maxDCG = torch.sum(Y.squeeze() / torch.log2(torch.argsort(torch.argsort(Y.squeeze(), descending=True)).float() + 2)).to(device)

            N = 1.0 / maxDCG
            # print("N just when create", N)
            # print("maxDCG", maxDCG)
            # input()
        # print("batch_counter", batch_counter, end="\r")
        batch_counter += 1
        # X_tensor = torch.tensor(X, dtype=precision, device=device)
        y_pred = net[rand_bool](X)
        y_pred_batch.append(y_pred)
        # compute the rank order of each document
        rank_order = torch.argsort(torch.argsort(Y.squeeze(), descending=True)) + 1

        with torch.no_grad():
            pos_pairs_score_diff = 1.0 + torch.exp(sigma * (y_pred - y_pred.t()))

            Y_tensor = Y
            rel_diff = Y_tensor - Y_tensor.t()
            pos_pairs = (rel_diff > 0).float()
            neg_pairs = (rel_diff < 0).float()
            # Hier aanpassen voor @5?
            
            Sij = pos_pairs - neg_pairs
            gain_diff = Y_tensor - Y_tensor.t()

            # print("gain_diff")
            # print(gain_diff)

            rank_order_tensor = torch.tensor(rank_order, dtype=float).view(-1, 1).to(device)
            decay_diff = 1.0 / torch.log2(rank_order_tensor + 1.0) - 1.0 / torch.log2(rank_order_tensor.t() + 1.0)
            # print("decay_diff")
            # print(decay_diff)

            # ALTERNATIVE
            # delta_ndcg = N * torch.abs((gain_diff * decay_diff) * (1 / pos_pairs_score_diff))
            # lambda_update = sigma  / pos_pairs_score_diff * delta_ndcg

            # ORIGINAL FROM GITHUB PAGE
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            lambda_update = sigma * (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg


            # print("N")
            # print(N)
            # print("torch.abs((gain_diff * decay_diff)")
            # print(torch.abs((gain_diff * decay_diff)))
            # print("delta_ndcg")
            # print(delta_ndcg)
            # print("(1 / pos_pairs_score_diff)")
            # print((1 / pos_pairs_score_diff))



            # print("lambda_update v1")
            # print(lambda_update)

            lambda_update = torch.sum(lambda_update, 1, keepdim=True)
            # print("lambda_update v2")
            # print(lambda_update)

            # input()

            assert lambda_update.shape == y_pred.shape
            grad_batch.append(lambda_update)

        # optimization is to similar to RankNetListWise, but to maximize NDCG.
        # lambda_update scales with gain and decay

        count += 1
        if count % batch_size == 0:
            for grad, y_pred in zip(grad_batch, y_pred_batch):
                # if grad.sum():
                #     print(" haeleluja we habben ne gradient:")
                #     print(grad)
                # print("gradient:")
                # print(grad)
                y_pred.backward(grad / batch_size)

            # print(net.hidden[0].weight)
            torch.nn.utils.clip_grad_norm_(net[rand_bool].parameters(), 10)
            optimizer[rand_bool].step()
            net[rand_bool].zero_grad()
            grad_batch, y_pred_batch = [], []  # grad_batch, y_pred_batch used for gradient_acc
    return net

    # optimizer.step()
    # eval_ndcg_at_k(net, device, df_train, train_loader, 100000, [10, 30, 50])
