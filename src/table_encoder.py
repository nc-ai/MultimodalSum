import torch
import torch.nn as nn


class YelpTableEncoder(nn.Module):
    def __init__(self, bart_embedding):
        super(YelpTableEncoder, self).__init__()
        self.bart_embedding = bart_embedding
        self.rating_embedding = nn.Linear(4, 1024, bias=False)
        self.hours_embedding = nn.Linear(4, 1024, bias=False)
        self.fc = nn.Linear(2048, 1024)
        self.linear = nn.Linear(1024, 1024, bias=False)

    def forward(self,
                field,
                field_value
               ):
        r'''
            field : tokenized field names [47, 6]
            field_value : list of tokenized field values
                name : [bsz, 24] # 1 field
                category : [bsz, 6, 12] # 1 field
                str_categorical : [bsz, 5, 3] # 5 fields
                str_boolean : [bsz, 32, 1] # 32 fields
                rating : [bsz, 4] # 1 field
                hours : [bsz, 7, 4] # 7 fields
        '''
        with torch.no_grad():
            field_name = self.bart_embedding(field) # [47, 6, 1024]
            field_mask = field.ne(1).unsqueeze(-1).float() # [47, 6, 1]
            field_name = torch.mul(field_name, field_mask).sum(dim=1)

        [name, category, str_categorical, str_boolean, rating, hours] = field_value

        # name (1) -> notnull
        with torch.no_grad():
            name_embedding = self.bart_embedding(name) # [bsz, 24, 1024]
            name_mask = name.ne(1).unsqueeze(-1).float() # [bsz, 24, 1]
            name_embedding = torch.mul(name_embedding, name_mask).sum(dim=1, keepdims=True)

        # category (1)
        with torch.no_grad():
            category_embedding = self.bart_embedding(category) # [bsz, 6, 12, 1024]
            category_mask = category.ne(1).unsqueeze(-1).float() # [bsz, 6, 12, 1]
            category_embedding = torch.mul(category_embedding, category_mask).sum(dim=2)
            category_mask = category.ne(1).max(dim=-1)[0].unsqueeze(-1).float() # [bsz, 6, 1]
            category_embedding = torch.mul(category_embedding, category_mask).sum(dim=1, keepdims=True) / (category_mask.sum(dim=1, keepdims=True) + 1e-6) # [bsz, 1, 1024]

        # str_categorical (5)
        with torch.no_grad():
            str_categorical_embedding = self.bart_embedding(str_categorical) # [bsz, 5, 3, 1024]
            str_categorical_mask = str_categorical.ne(1).unsqueeze(-1).float() # [bsz, 5, 3, 1]
            str_categorical_embedding = torch.mul(str_categorical_embedding, str_categorical_mask).sum(dim=2)

        # str_boolean (32)
        with torch.no_grad():
            str_boolean_embedding = self.bart_embedding(str_boolean.squeeze(-1)) # [bsz, 32, 1024]
            str_boolean_mask = str_boolean.ne(1).float() # [bsz, 32, 1]
            str_boolean_embedding = torch.mul(str_boolean_embedding, str_boolean_mask)

        # rating (1) -> notnull
        rating_embedding = self.rating_embedding(rating.float()).unsqueeze(1) # [bsz, 1, 1024]

        # hours (7)
        hours_embedding = self.hours_embedding(hours.float()) # [bsz, 7, 1024]

        all_names = field_name.unsqueeze(0).repeat([name.size(0), 1, 1]) # [bsz, 47, 1024]
        all_values = torch.cat([name_embedding, category_embedding, str_categorical_embedding, str_boolean_embedding, rating_embedding, hours_embedding], dim=1) # [bsz, 47, 1024]
        all_embeddings = torch.cat([all_names, all_values], dim=-1) # [bsz, 47, 2048]

        all_embeddings = self.fc(all_embeddings) # [bsz, 47, 1024]
        all_embeddings = torch.relu(all_embeddings)
        all_embeddings = self.linear(all_embeddings)

        with torch.no_grad():
            name_mask = torch.tensor([[True]] * name.size(0), device=name.device) # [bsz, 1]
            category_mask = category[:, :1, 0].ne(1) # [bsz, 1]
            str_categorical_mask = str_categorical[:, :, 0].ne(1) # [bsz, 5]
            str_boolean_mask = str_boolean[:, :, 0].ne(1) # [bsz, 32]
            rating_mask = torch.tensor([[True]] * name.size(0), device=name.device) # [bsz, 1]
            hours_mask = hours.sum(dim=-1, keepdims=False) != 0. # [bsz, 7]
            all_masks = torch.cat([name_mask, category_mask, str_categorical_mask, str_boolean_mask, rating_mask, hours_mask], dim=1) # [bsz, 47]
        return all_embeddings, all_masks


class AmazonTableEncoder(nn.Module):
    def __init__(self, bart_embedding):
        super(AmazonTableEncoder, self).__init__()
        self.bart_embedding = bart_embedding
        self.price_embedding = nn.Linear(11, 1024, bias=False)
        self.rating_embedding = nn.Linear(4, 1024, bias=False)
        self.fc = nn.Linear(2048, 1024)
        self.linear = nn.Linear(1024, 1024, bias=False)

    def forward(self,
                field,
                field_value
               ):
        r'''
            field_name : tokenized field names [6, 1]
            field_value : list of tokenized field values
                price : [bsz, 11] # 1 field
                rating : [bsz, 4] # 1 field
                brand : [bsz, 12] # 1 field
                name : [bsz, 32] # 1 field
                category : [bsz, 3, 8, 12] # 1 field
                description : [bsz, 128] # 128 fields
        '''
        with torch.no_grad():
            field_name = self.bart_embedding(field).squeeze(1) # [6, 1024]
            field_name = torch.cat([field_name[:-1, :], field_name[-1:, :].repeat([128, 1])]) # [5+128, 1024]

        [price, rating, brand, name, category, description] = field_value

        # price & rating
        price_embedding = self.price_embedding(price.float()).unsqueeze(1) # [bsz, 1, 1024]
        rating_embedding = self.rating_embedding(rating.float()).unsqueeze(1) # [bsz, 1, 1024]

        # brand
        with torch.no_grad():
            brand_embedding = self.bart_embedding(brand) # [bsz, 12, 1024]
            brand_mask = brand.ne(1).unsqueeze(-1).float() # [bsz, 12, 1]
            brand_embedding = torch.mul(brand_embedding, brand_mask).sum(dim=1, keepdims=True) # [bsz, 1, 1024]

        # name
        with torch.no_grad():
            name_embedding = self.bart_embedding(name) # [bsz, 32, 1024]
            name_mask = name.ne(1).unsqueeze(-1).float() # [bsz, 32, 1]
            name_embedding = torch.mul(name_embedding, name_mask).sum(dim=1, keepdims=True) # [bsz, 1, 1024]

        # category
        with torch.no_grad():
            category_embedding = self.bart_embedding(category) # [bsz, 3, 8, 12, 1024]

            category_mask_ = category.ne(1) # [bsz, 3, 8, 12]
            category_mask = category_mask_.unsqueeze(-1).float() # [bsz, 3, 8, 12, 1]
            category_embedding = torch.mul(category_embedding, category_mask).sum(dim=3) # [bsz, 3, 8, 1024]

            category_mask_ = category_mask_.max(dim=-1)[0] # [bsz, 3, 8]
            category_mask = category_mask_.unsqueeze(-1).float() # [bsz, 3, 8, 1]
            category_embedding = torch.mul(category_embedding, category_mask).sum(dim=2) / (category_mask.sum(dim=2) + 1e-6) # [bsz, 3, 1024]

            category_mask_ = category_mask_.max(dim=-1)[0] # [bsz, 3]
            category_mask = category_mask_.unsqueeze(-1).float() # [bsz, 3, 1]
            category_embedding = torch.mul(category_embedding, category_mask).sum(dim=1, keepdims=True) / (category_mask.sum(dim=1, keepdims=True) + 1e-6) # [bsz, 1, 1024]

        # description
        with torch.no_grad():
            description_embedding = self.bart_embedding(description) # [bsz, 128, 1024]
            description_mask = description.ne(1) # [bsz, 128]

        all_names = field_name.unsqueeze(0).repeat([price.size(0), 1, 1]) # [bsz, 5+128, 1024]
        all_values = torch.cat([price_embedding, rating_embedding, brand_embedding, name_embedding, category_embedding, description_embedding], dim=1) # [bsz, 5+128, 1024]
        all_embeddings = torch.cat([all_names, all_values], dim=-1) # [bsz, 5+128, 2048]

        all_embeddings = self.fc(all_embeddings) # [bsz, 5+128, 1024]
        all_embeddings = torch.relu(all_embeddings)
        all_embeddings = self.linear(all_embeddings)

        with torch.no_grad():
            price_mask = price.sum(dim=1, keepdims=True) != 0. # [bsz, 1]
            rating_mask = torch.tensor([[True]] * price.size(0), device=price.device) # [bsz, 1]
            brand_mask = brand[:, :1].ne(1) # [bsz, 1]
            name_mask = name[:, :1].ne(1) # [bsz, 1]
            category_mask = torch.tensor([[True]] * price.size(0), device=price.device) # [bsz, 1]
            all_masks = torch.cat([price_mask, rating_mask, brand_mask, name_mask, category_mask, description_mask], dim=1) # [bsz, 5+128]
        return all_embeddings, all_masks