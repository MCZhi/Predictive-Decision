import torch
import math
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe

        return self.dropout(x)

class CrossTransformer(nn.Module):
    def __init__(self):
        super(CrossTransformer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(128, 8, 0.1, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 128), nn.LayerNorm(128))

    def forward(self, query, key, mask=None):
        value = key
        mask[:, 0] = False
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        output = self.ffn(attention_output)

        return output

class SelfTransformer(nn.Module):
    def __init__(self):
        super(SelfTransformer, self).__init__()
        self.self_attention = nn.MultiheadAttention(128, 8, 0.1, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 128), nn.LayerNorm(128))

    def forward(self, input, mask=None):
        attention_output, _ = self.self_attention(input, input, input, key_padding_mask=mask)
        output = self.ffn(attention_output)

        return output

class AgentEncoder(nn.Module):
    def __init__(self):
        super(AgentEncoder, self).__init__()
        self.position = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 128))
        self.encode = PositionalEncoding(d_model=128, max_len=11)
        self.history = SelfTransformer()

    def forward(self, inputs):
        mask = torch.eq(inputs[:, :, 0], 0)
        mask[:, -1] = False 
        time = self.history(self.encode(self.position(inputs)), mask=mask)
        output = time[:, -1]

        return output

class MapEncoder(nn.Module):
    def __init__(self):
        super(MapEncoder, self).__init__()
        self.waypoint = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 128))

    def forward(self, inputs):
        output = self.waypoint(inputs)

        return output

class Agent2Agent(nn.Module):
    def __init__(self):
        super(Agent2Agent, self).__init__()
        self.interaction_1 = SelfTransformer()
        self.interaction_2 = SelfTransformer()

    def forward(self, inputs, mask=None):
        output = self.interaction_1(inputs, mask=mask)
        output = self.interaction_2(inputs+output, mask=mask)

        return output

class Agent2Map(nn.Module):
    def __init__(self):
        super(Agent2Map, self).__init__()
        self.position_encode = PositionalEncoding(d_model=128, max_len=51)
        self.lane = CrossTransformer()
        self.map = CrossTransformer()

    def forward(self, actor, waypoints, mask):
        query = actor.unsqueeze(1)
        lane_attention = torch.cat([self.lane(query, self.position_encode(waypoints[:, i]), mask[:, i]) 
                                    for i in range(waypoints.shape[1])], dim=1)
        map_attention = self.map(query, lane_attention, mask[:, :, 10])
        output = map_attention.squeeze(1)

        return output

class Decoder(nn.Module):
    def __init__(self, use_interaction):
        super(Decoder, self).__init__()
        self.use_interaction = use_interaction
        if use_interaction:
            self.cell = nn.GRUCell(input_size=128, hidden_size=384)
            self.plan_input = nn.Linear(3, 128)
            self.state_input = nn.Linear(3, 128)
        else:
            self.cell = nn.GRUCell(input_size=3, hidden_size=384)
        self.decode = nn.Sequential(nn.Dropout(0.1), nn.Linear(384, 64), nn.ELU(), nn.Linear(64, 3))

    def forward(self, init_hidden, plan, gate, init_state):
        output = []
        hidden = init_hidden
        state = init_state

        for t in range(30):
            if self.use_interaction:
                plan_input = self.plan_input(plan[:, t, :3]) 
                state_input = self.state_input(state[:, :3])
                input = state_input + plan_input * gate
            else:
                input = state[:, :3]

            hidden = self.cell(input, hidden)
            state = self.decode(hidden) + state[:, :3]
            output.append(state)

        output = torch.stack(output, dim=1)

        return output

class Predictor(nn.Module):
    def __init__(self, use_interaction):
        super(Predictor, self).__init__()
        # Observation space
        # Ego: (B, T_h, 4)
        # Neighbor: (B, N_n, T_h, 4)
        # Ego map: (B, N_l, 51, 4)
        # Neighbor map: (B, N_n, N_l, 51, 4) 
        # Plan: (B, T_f, 4)
        
        # agent layer
        self.ego_net = AgentEncoder()
        self.neighbor_net = AgentEncoder()

        # map layer
        self.map_net = MapEncoder()
        
        # attention layers
        self.agent_map = Agent2Map()
        self.agent_agent = Agent2Agent()
        self.gate = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

        # decoder layer
        self.decoder = Decoder(use_interaction)
       
    def forward(self, observations, plan):
        # get inputs and encode them
        for key, sub_space in observations.items():
            if key == 'ego_state':
                ego = sub_space
                encoded_ego = [self.ego_net(ego)]
            elif key == 'neighbors_state':
                neighbors = sub_space
                encoded_neighbors = [self.neighbor_net(neighbors[:, i]) for i in range(neighbors.shape[1])]
            elif key == 'ego_map':
                ego_map = sub_space
                encoded_ego_map = self.map_net(ego_map)
            elif key == 'neighbors_map':
                neighbor_map = sub_space
                encoded_neighbor_map = self.map_net(neighbor_map)
            else:
                raise KeyError
                
        # agent-agent interaction Transformer
        encoded_actors = torch.stack(encoded_ego + encoded_neighbors, dim=1)
        actor_mask = torch.eq(torch.cat([ego.unsqueeze(1), neighbors], dim=1), 0)[:, :, -1, 0]
        actor_mask[:, 0] = False
        agent_agent = self.agent_agent(encoded_actors, actor_mask)

        # agent-map Transformer
        per_agent_tensor_list = []
        for i in range(neighbors.shape[1]):
            map_mask = torch.eq(neighbor_map[:, i, :, :, -1], 0)
            agent_map = self.agent_map(agent_agent[:, i+1], encoded_neighbor_map[:, i], map_mask)
            per_agent_tensor_list.append(torch.cat([agent_map, encoded_neighbors[i], agent_agent[:, i+1]], dim=-1))

        # decode interaction-aware trajectories
        per_agent_prediction_list = []
        for i in range(neighbors.shape[1]):
            gate = self.gate(torch.cat([encoded_ego[0], encoded_neighbors[i]], dim=-1))
            predict_traj = self.decoder(per_agent_tensor_list[i], plan, gate, neighbors[:, i, -1])
            per_agent_prediction_list.append(predict_traj)

        prediction = torch.stack(per_agent_prediction_list, dim=1)

        return prediction
