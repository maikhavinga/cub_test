import pandas as pd


def create_category_dict():
    status_current_account_dict = {"A11": "Less_than_0",
                                   "A12": "Between_0_and_200",
                                   "A13": "Higher_than_200",
                                   "A14": "No_checking_account"}
    credit_history_dict = {"A30": "No_credits_taken",
                           "A31": "All_credits_paid_duly",
                           "A32": "Existing_credits_paid_duly_so_far",
                           "A33": "Delay_paying_off_in_past",
                           "A34": "Critical_account"}
    purpose_dict = {"A40": "Car_new",
                    "A41": "Car_old",
                    "A42": "Furniture_equipment",
                    "A43": "Radio_television",
                    "A44": "Domestic_appliance",
                    "A45": "Repairs",
                    "A46": "Education",
                    "A48": "Retraining",
                    "A49": "Business",
                    "A410": "Others"}
    savings_dict = {"A61": "Less_than_100",
                    "A62": "Between_100_and_500",
                    "A63": "Between_500_and_1000",
                    "A64": "More_than_1000",
                    "A65": "Unknown_or_no_savings_account"}
    employment_status_dict = {"A71": "Unemployed",
                              "A72": "Less_than_1_year",
                              "A73": "Between_1_and_4_years",
                              "A74": "Between_4_and_7_years",
                              "A75": "More_than_7_years"}
    personal_status_dict = {"A91": "Male_divorced_seperated",
                            "A92": "Female_divorced_seperated_married",
                            "A93": "Male_single",
                            "A94": "Male_married_widowed",
                            "A95": "Female_single"}
    other_debtors_dict = {"A101": "None",
                          "A102": "Co-applicant",
                          "A103": "Guarantor"}
    property_dict = {"A121": "Real_estate",
                     "A122": "Building_society_life_insurance",
                     "A123": "Car_or_other",
                     "A124": "Unknown_no_property"}
    other_installment_plans_dict = {"A141": "Bank",
                                    "A142": "Stores",
                                    "A143": "None"}
    housing_dict = {"A151": "Rent",
                    "A152": "Own",
                    "A153": "For_free"}
    job_dict = {"A171": "Unemployed_unskilled_non_resident",
                "A172": "Unskilled_resident",
                "A173": "Skilled_employee",
                "A174": "Management_self_employed"}
    telephone_dict = {"A191": "None",
                      "A192": "Yes"}
    foreign_worker_dict = {"A201": "Yes",
                           "A202": "No"}
    return {"Status_current_account": status_current_account_dict,
            "Credit_history": credit_history_dict,
            "Purpose": purpose_dict,
            "Savings": savings_dict,
            "Employment_status": employment_status_dict,
            "Personal_status": personal_status_dict,
            "Other_debtors": other_debtors_dict,
            "Property": property_dict,
            "Other_installment_plans": other_installment_plans_dict,
            "Housing": housing_dict,
            "Job": job_dict,
            "Telephone": telephone_dict,
            "Foreign_worker": foreign_worker_dict}


def categorize_data(data):
    category_dict = create_category_dict()
    for column in data.columns:
        if column in category_dict:
            data[column] = data[column].map(category_dict[column]).astype("category")
    return data


def int_to_float_data(data):
    category_dict = create_category_dict()
    for column in data.columns:
        if column not in category_dict:
            data[column] = data[column].astype("float")
    return data


def transform_target(data):
    data['Good_credit'] = 2 - data['Good_credit']
    return data
