class Special_Tokens:
  def __init__(self):
    self.special_tokens = ['[attraction_address]',
     '[attraction_name]',
     '[attraction_phone]',
     '[attraction_postcode]',
     '[attraction_reference]',
     '[hospital_address]',
     '[hospital_department]',
     '[hospital_name]',
     '[hospital_phone]',
     '[hospital_postcode]',
     '[hospital_reference]',
     '[hotel_address]',
     '[hotel_name]',
     '[hotel_phone]',
     '[hotel_postcode]',
     '[hotel_reference]',
     '[police_address]',
     '[police_phone]',
     '[police_postcode]',
     '[restaurant_address]',
     '[restaurant_name]',
     '[restaurant_phone]',
     '[restaurant_postcode]',
     '[restaurant_reference]',
     '[taxi_phone]',
     '[taxi_type]',
     '[train_id]',
     '[train_reference]',
     '[value_area]',
     '[value_count]',
     '[value_day]',
     '[value_food]',
     '[value_place]',
     '[value_price]',
     '[value_pricerange]',
     '[value_time]']


if __name__ == '__main__':
    tokens = Special_Tokens()
    print(tokens.special_tokens)