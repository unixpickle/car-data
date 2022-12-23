PRICE_CUTOFFS = [
    10_000.0,
    15_000.0,
    20_000.0,
    25_000.0,
    30_000.0,
    35_000.0,
    40_000.0,
    50_000.0,
    60_000.0,
]

NUM_PRICE_BINS = len(PRICE_CUTOFFS) + 1

PRICE_BIN_LABELS = [
    "$0-$10,000",
    "$10,000-$15,000",
    "$15,000-$20,000",
    "$20,000-$25,000",
    "$25,000-$30,000",
    "$30,000-$35,000",
    "$35,000-$40,000",
    "$40,000-$50,000",
    "$50,000-$60,000",
    "$60,000+",
]

MEDIAN_PRICE_SCALE = 30000.0

YEARS = list(range(1983, 2024))
NUM_YEARS = len(YEARS) + 1

MAKES_MODELS = (
    ("Ford", "F150"),
    ("Chevrolet", "Silverado 1500"),
    ("RAM", "1500"),
    ("Jeep", "Wrangler"),
    ("Ford", "Explorer"),
    ("Nissan", "Rogue"),
    ("Jeep", "Grand Cherokee"),
    ("Chevrolet", "Equinox"),
    ("GMC", "Sierra 1500"),
    ("Ford", "Escape"),
    ("Honda", "Accord"),
    ("Toyota", "Camry"),
    ("Toyota", "RAV4"),
    ("Honda", "Civic"),
    ("Honda", "CR-V"),
    ("MAZDA", "CX-5"),
    ("Toyota", "Tacoma"),
    ("Ford", "F250"),
    ("Toyota", "Corolla"),
    ("Toyota", "Highlander"),
    ("Jeep", "Cherokee"),
    ("Nissan", "Altima"),
    ("Subaru", "Outback"),
    ("RAM", "2500"),
    ("Honda", "Pilot"),
    ("Chevrolet", "Malibu"),
    ("Hyundai", "Tucson"),
    ("Ford", "Mustang"),
    ("Chevrolet", "Traverse"),
    ("Hyundai", "Santa Fe"),
    ("Hyundai", "Elantra"),
    ("Jeep", "Compass"),
    ("Chevrolet", "Silverado 2500"),
    ("Ford", "Edge"),
    ("Nissan", "Frontier"),
    ("Chevrolet", "Tahoe"),
    ("GMC", "Terrain"),
    ("Toyota", "Tundra"),
    ("GMC", "Acadia"),
    ("Volkswagen", "Tiguan"),
    ("Hyundai", "Sonata"),
    ("Subaru", "Forester"),
    ("Jeep", "Gladiator"),
    ("Chevrolet", "Colorado"),
    ("Nissan", "Pathfinder"),
    ("Toyota", "4Runner"),
    ("Ford", "Fusion"),
    ("Nissan", "Sentra"),
    ("Kia", "Sorento"),
    ("GMC", "Sierra 2500"),
    ("Ford", "F350"),
    ("Subaru", "Crosstrek"),
    ("Kia", "Sportage"),
    ("Honda", "HR-V"),
    ("Kia", "Forte"),
    ("Honda", "Odyssey"),
    ("Ford", "Bronco Sport"),
    ("Dodge", "Challenger"),
    ("Dodge", "Charger"),
    ("Buick", "Enclave"),
    ("Chevrolet", "Blazer"),
    ("Acura", "MDX"),
    ("Audi", "Q5"),
    ("Volkswagen", "Atlas"),
    ("Buick", "Envision"),
    ("Kia", "Soul"),
    ("Chrysler", "Pacifica"),
    ("Hyundai", "Kona"),
    ("Chevrolet", "Camaro"),
    ("Jeep", "Grand Cherokee L"),
    ("MAZDA", "CX-9"),
    ("Dodge", "Durango"),
    ("Nissan", "Murano"),
    ("Chevrolet", "Trax"),
    ("GMC", "Yukon"),
    ("Volkswagen", "Jetta"),
    ("BMW", "X5"),
    ("Chevrolet", "Suburban"),
    ("Ford", "Expedition"),
    ("Nissan", "Rogue Sport"),
    ("RAM", "3500"),
    ("Ford", "Bronco"),
    ("Honda", "Ridgeline"),
    ("Chevrolet", "Corvette"),
    ("Cadillac", "XT5"),
    ("Toyota", "Sienna"),
    ("Mitsubishi", "Outlander"),
    ("Kia", "Telluride"),
    ("Buick", "Encore"),
    ("Mercedes-Benz", "C 300"),
    ("BMW", "X3"),
    ("Subaru", "Ascent"),
    ("Honda", "Passport"),
    ("MAZDA", "MAZDA3"),
    ("Buick", "Encore GX"),
    ("Volvo", "XC90"),
    ("Mercedes-Benz", "GLC 300"),
    ("Ford", "Ranger"),
    ("Jeep", "Renegade"),
    ("Lexus", "RX 350"),
    ("Volvo", "XC60"),
    ("Kia", "Optima"),
    ("Chevrolet", "Silverado 3500"),
    ("Dodge", "Grand Caravan"),
    ("INFINITI", "QX60"),
    ("Nissan", "Titan"),
    ("Subaru", "WRX"),
    ("GMC", "Canyon"),
    ("Tesla", "Model 3"),
    ("Chevrolet", "Cruze"),
    ("Lexus", "ES 350"),
    ("Nissan", "Armada"),
    ("GMC", "Yukon XL"),
    ("GMC", "Sierra 3500"),
    ("Hyundai", "Palisade"),
    ("Ford", "Focus"),
    ("Kia", "Niro"),
    ("Toyota", "Prius"),
    ("INFINITI", "QX80"),
    ("Porsche", "Macan"),
    ("Chevrolet", "TrailBlazer"),
    ("Cadillac", "XT4"),
    ("MAZDA", "CX-50"),
    ("Lincoln", "Corsair"),
    ("Audi", "Q7"),
    ("Ford", "Expedition Max"),
    ("Cadillac", "Escalade"),
    ("MINI", "Cooper"),
    ("Acura", "RDX"),
    ("Subaru", "Impreza"),
    ("Audi", "A4"),
    ("Nissan", "Kicks"),
    ("Nissan", "Maxima"),
    ("Porsche", "Cayenne"),
    ("Dodge", "Journey"),
    ("Porsche", "911"),
    ("RAM", "ProMaster"),
    ("Mercedes-Benz", "GLE 350"),
    ("Ford", "EcoSport"),
    ("Volkswagen", "Taos"),
    ("MAZDA", "CX-30"),
    ("Lincoln", "Nautilus"),
    ("Land Rover", "Range Rover"),
    ("Mitsubishi", "Outlander Sport"),
    ("Lexus", "GX 460"),
    ("Volkswagen", "Passat"),
    ("Land Rover", "Range Rover Sport"),
    ("Nissan", "Versa"),
    ("Volvo", "XC40"),
    ("Mercedes-Benz", "E 350"),
    ("Chrysler", "300"),
    ("Chevrolet", "Impala"),
    ("Subaru", "Legacy"),
    ("Acura", "TLX"),
    ("Mercedes-Benz", "Sprinter"),
    ("Cadillac", "CT5"),
    ("Mercedes-Benz", "GLA 250"),
    ("Hyundai", "Santa Cruz"),
    ("Tesla", "Model S"),
    ("Mercedes-Benz", "GLB 250"),
    ("INFINITI", "Q50"),
    ("Kia", "K5"),
    ("Cadillac", "XT6"),
    ("Audi", "Q3"),
    ("INFINITI", "QX50"),
    ("Ford", "Transit 250"),
    ("Ford", "Mustang Mach-E"),
    ("Kia", "Seltos"),
    ("MAZDA", "MX-5 Miata"),
    ("Audi", "A5"),
    ("Lincoln", "Aviator"),
    ("BMW", "X1"),
    ("Kia", "Rio"),
    ("Chevrolet", "Express 2500"),
    ("Ford", "Transit 350"),
    ("Toyota", "Venza"),
    ("Mercedes-Benz", "S 500"),
    ("Cadillac", "Escalade ESV"),
    ("Jeep", "Wagoneer"),
    ("Chevrolet", "Bolt"),
    ("MINI", "Cooper Countryman"),
    ("Toyota", "Sequoia"),
    ("Mercedes-Benz", "CLA 250"),
    ("BMW", "X7"),
    ("Cadillac", "CTS"),
    ("Hyundai", "Venue"),
    ("Volkswagen", "ID.4"),
    ("Toyota", "Avalon"),
    ("Jeep", "Patriot"),
    ("Tesla", "Model Y"),
    ("Nissan", "Leaf"),
    ("Audi", "A3"),
    ("Acura", "Integra"),
    ("Ford", "Transit Connect"),
    ("Lexus", "NX 300"),
    ("Audi", "A6"),
    ("Mercedes-Benz", "EQS 450+"),
    ("Chevrolet", "Spark"),
    ("Jaguar", "F-PACE"),
    ("Mercedes-Benz", "S 580"),
    ("Chevrolet", "Sonic"),
    ("Lincoln", "Navigator"),
    ("Toyota", "C-HR"),
    ("Ford", "Fiesta"),
    ("RAM", "ProMaster City"),
    ("Volvo", "S60"),
    ("BMW", "330i xDrive"),
    ("Ford", "Flex"),
    ("MAZDA", "MAZDA6"),
    ("Toyota", "Corolla Cross"),
    ("Lincoln", "MKZ"),
    ("Chevrolet", "Express 3500"),
    ("Hyundai", "Accent"),
    ("Land Rover", "Discovery Sport"),
    ("Tesla", "Model X"),
    ("Honda", "Fit"),
    ("Alfa Romeo", "Stelvio"),
    ("Chrysler", "200"),
    ("Volkswagen", "Beetle"),
    ("Cadillac", "CT4"),
    ("Ford", "Maverick"),
    ("Volkswagen", "GTI"),
    ("Lincoln", "MKC"),
    ("Porsche", "Panamera"),
    ("Ford", "F450"),
    ("Lexus", "NX 350"),
    ("Chrysler", "Town & Country"),
    ("Kia", "Stinger"),
    ("Land Rover", "Range Rover Velar"),
    ("Audi", "S5"),
    ("BMW", "330i"),
    ("Volkswagen", "Golf"),
    ("Mercedes-Benz", "GLS 450"),
    ("Lexus", "IS 350"),
    ("Land Rover", "Range Rover Evoque"),
    ("Toyota", "Prius Prime"),
    ("Acura", "ILX"),
    ("Genesis", "G70"),
    ("Ford", "Taurus"),
    ("Hyundai", "Veloster"),
    ("Lexus", "IS 300"),
    ("Land Rover", "Defender"),
    ("Genesis", "GV80"),
    ("Alfa Romeo", "Giulia"),
    ("BMW", "X6"),
    ("Hyundai", "Ioniq 5"),
    ("Audi", "SQ5"),
    ("BMW", "328i"),
    ("BMW", "i3"),
    ("Cadillac", "ATS"),
    ("Mercedes-Benz", "S 550"),
    ("Lincoln", "Navigator L"),
    ("Mercedes-Benz", "E 450"),
    ("Buick", "LaCrosse"),
    ("Ford", "E-350 and Econoline 350"),
    ("BMW", "M3"),
    ("Mercedes-Benz", "GLE 53 AMG"),
    ("Lexus", "IS 250"),
    ("Mercedes-Benz", "E 300"),
    ("Cadillac", "SRX"),
    ("GMC", "Savana 2500"),
    ("INFINITI", "QX55"),
    ("Mitsubishi", "Eclipse Cross"),
    ("Audi", "Q8"),
    ("INFINITI", "Q60"),
    ("Kia", "Sedona"),
    ("Lincoln", "MKX"),
    ("Audi", "e-tron"),
    ("Chevrolet", "Volt"),
    ("BMW", "X4"),
    ("Chevrolet", "Bolt EUV"),
    ("Volvo", "C40"),
    ("Maserati", "Ghibli"),
    ("Lexus", "ES 300h"),
    ("Jaguar", "F-TYPE"),
    ("Cadillac", "XTS"),
    ("Genesis", "GV70"),
    ("BMW", "430i xDrive"),
    ("BMW", "430i"),
    ("BMW", "Z4"),
    ("BMW", "M4"),
    ("Land Rover", "Discovery"),
    ("Lexus", "GS 350"),
    ("Mercedes-Benz", "A 220"),
    ("Dodge", "Ram 1500 Truck"),
    ("Ford", "F550"),
    ("Hyundai", "Ioniq"),
    ("Mercedes-Benz", "ML 350"),
    ("Genesis", "G80"),
    ("MINI", "Cooper Clubman"),
    ("Maserati", "Levante"),
    ("Mercedes-Benz", "AMG GT"),
    ("BMW", "530i xDrive"),
    ("Lincoln", "Continental"),
    ("Chrysler", "Voyager"),
    ("Lexus", "LS 460"),
    ("MAZDA", "MX-5 Miata RF"),
    ("FIAT", "500"),
    ("Cadillac", "CT6"),
    ("MAZDA", "CX-3"),
    ("BMW", "M5"),
    ("BMW", "328i xDrive"),
    ("Hyundai", "Genesis"),
    ("Kia", "EV6"),
    ("INFINITI", "G37"),
    ("Audi", "A8"),
    ("Audi", "S4"),
    ("BMW", "X2"),
    ("BMW", "530i"),
    ("Lexus", "UX 250h"),
    ("Lexus", "RX 350L"),
    ("Mercedes-Benz", "G 63 AMG"),
    ("Nissan", "Juke"),
    ("Volkswagen", "Arteon"),
    ("Honda", "Insight"),
    ("Lexus", "RC 350"),
    ("RAM", "5500"),
    ("Audi", "A7"),
    ("Lexus", "NX 200t"),
    ("Nissan", "370Z"),
    ("Porsche", "Boxster"),
    ("BMW", "540i"),
    ("Buick", "Regal"),
    ("Dodge", "Dart"),
    ("BMW", "540i xDrive"),
    ("Mercedes-Benz", "GLE 450"),
    ("Ford", "Expedition EL"),
    ("Jeep", "Grand Wagoneer"),
    ("Bentley", "Continental"),
    ("Dodge", "Ram 2500 Truck"),
    ("Jeep", "Liberty"),
    ("Kia", "Carnival"),
    ("Mitsubishi", "Mirage G4"),
    ("Mercedes-Benz", "GL 450"),
    ("Mitsubishi", "Mirage"),
    ("Lexus", "RX 450h"),
    ("Porsche", "Taycan"),
    ("Acura", "TL"),
    ("Lexus", "CT 200h"),
    ("Nissan", "NV"),
    ("BMW", "440i xDrive"),
    ("Mercedes-Benz", "C 43 AMG"),
    ("Mercedes-Benz", "EQS 580"),
    ("Toyota", "Supra"),
    ("Mercedes-Benz", "GLK 350"),
    ("Lexus", "LS 500"),
    ("Toyota", "Prius C"),
    ("Toyota", "Yaris"),
    ("Jaguar", "XF"),
    ("Nissan", "Versa Note"),
    ("BMW", "335i"),
    ("Nissan", "Xterra"),
    ("Lexus", "NX 250"),
    ("Toyota", "FJ Cruiser"),
    ("Audi", "RS 5"),
    ("Volvo", "V60"),
    ("Audi", "S3"),
    ("BMW", "740i"),
    ("BMW", "128i"),
    ("Buick", "Verano"),
    ("Subaru", "BRZ"),
    ("Audi", "Q4 e-tron"),
    ("Chevrolet", "Avalanche"),
    ("Mercedes-Benz", "SL 550"),
    ("Ford", "C-MAX"),
    ("Toyota", "GR86"),
    ("BMW", "750i xDrive"),
    ("Ford", "Transit 150"),
    ("Mercedes-Benz", "Metris"),
    ("Mercedes-Benz", "S 560"),
    ("Nissan", "NV200"),
    ("Volkswagen", "Golf R"),
    ("Mercedes-Benz", "SL 63 AMG"),
    ("BMW", "M850i xDrive"),
    ("Lexus", "LX 570"),
    ("Mercedes-Benz", "G 550"),
    ("Ford", "E-450 and Econoline 450"),
    ("Ford", "E-Transit"),
    ("Mercedes-Benz", "C 250"),
    ("Mercedes-Benz", "CLS 450"),
    ("Mercedes-Benz", "S 63 AMG"),
    ("BMW", "530e"),
    ("BMW", "428i"),
    ("Mercedes-Benz", "GLC 43 AMG"),
    ("Volvo", "S90"),
    ("Dodge", "Avenger"),
    ("Lexus", "NX 300h"),
    ("Mercedes-Benz", "GLE 43 AMG"),
    ("Mercedes-Benz", "E 400"),
    ("Toyota", "Prius V"),
    ("BMW", "X5 M"),
    ("GMC", "Savana 3500"),
    ("Scion", "tC"),
    ("Volkswagen", "CC"),
    ("Acura", "TSX"),
    ("BMW", "228i xDrive"),
    ("BMW", "535i xDrive"),
    ("Porsche", "Cayman"),
    ("Subaru", "Impreza WRX"),
    ("BMW", "535i"),
    ("BMW", "M8"),
    ("Bentley", "Bentayga"),
    ("Maserati", "Quattroporte"),
    ("BMW", "M550i xDrive"),
    ("Jaguar", "XE"),
    ("Hyundai", "Kona N"),
    ("Porsche", "718 Cayman"),
    ("BMW", "M2"),
    ("Mercedes-Benz", "C 63 AMG"),
    ("BMW", "M340i"),
    ("Hyundai", "Elantra N"),
    ("BMW", "528i"),
    ("Ford", "E-250 and Econoline 250"),
    ("BMW", "i4"),
    ("FIAT", "500X"),
    ("BMW", "iX"),
    ("Audi", "TT"),
    ("Lexus", "IS 200t"),
    ("Maserati", "GranTurismo"),
    ("Dodge", "Ram 3500 Truck"),
    ("BMW", "650i"),
    ("Lexus", "UX 200"),
    ("Dodge", "Dakota"),
    ("INFINITI", "QX30"),
    ("Mercedes-Benz", "GLE 63 AMG"),
    ("Volkswagen", "Touareg"),
    ("Volkswagen", "e-Golf"),
    ("Lamborghini", "Huracan"),
    ("Lexus", "LC 500"),
    ("Land Rover", "LR4"),
    ("Lexus", "NX 350h"),
    ("BMW", "428i xDrive"),
    ("Jaguar", "XJ"),
    ("Lexus", "RC 300"),
    ("Toyota", "Mirai"),
    ("BMW", "330e"),
    ("Genesis", "G90"),
    ("Jaguar", "E-PACE"),
    ("Lamborghini", "Urus"),
    ("BMW", "M340i xDrive"),
    ("Audi", "RS 7"),
    ("Lexus", "ES 250"),
    ("Mercedes-Benz", "SL 55 AMG"),
    ("BMW", "320i"),
    ("Toyota", "Land Cruiser"),
    ("Ford", "Thunderbird"),
    ("Honda", "Element"),
    ("Scion", "xB"),
    ("BMW", "530e xDrive"),
    ("Porsche", "718 Boxster"),
    ("Buick", "Lucerne"),
    ("Mercedes-Benz", "E 53 AMG"),
    ("Mitsubishi", "Lancer"),
    ("Polestar", "Polestar 2"),
    ("RAM", "4500"),
    ("Scion", "FR-S"),
    ("Mercedes-Benz", "E 550"),
    ("Nissan", "GT-R"),
    ("BMW", "X6 M"),
    ("INFINITI", "Q70"),
    ("Audi", "R8"),
    ("Honda", "Clarity"),
    ("Mercedes-Benz", "E 63 AMG"),
    ("BMW", "320i xDrive"),
    ("Ford", "E-150 and Econoline 150"),
    ("Lexus", "GX 470"),
    ("Lincoln", "MKS"),
    ("BMW", "135i"),
    ("Mercedes-Benz", "GL 550"),
    ("Toyota", "86"),
    ("smart", "fortwo"),
    ("Chevrolet", "Express 1500"),
    ("BMW", "528i xDrive"),
    ("BMW", "M440i"),
    ("BMW", "230i"),
    ("INFINITI", "G35"),
    ("Mercedes-Benz", "S 450"),
    ("Mercedes-Benz", "SL 500"),
    ("BMW", "435i xDrive"),
    ("FIAT", "124 Spider"),
    ("Mercedes-Benz", "CLS 550"),
    ("Mercedes-Benz", "EQE 350+"),
    ("Mercury", "Grand Marquis"),
    ("Volkswagen", "Eos"),
    ("Chrysler", "PT Cruiser"),
    ("Lexus", "SC 430"),
    ("Lincoln", "Town Car"),
    ("Nissan", "Quest"),
    ("Audi", "S8"),
    ("BMW", "435i"),
    ("HUMMER", "H2"),
    ("Kia", "Cadenza"),
    ("BMW", "228i"),
    ("Chrysler", "Sebring"),
    ("Volvo", "XC70"),
    ("BMW", "335i xDrive"),
    ("Chevrolet", "Captiva Sport"),
    ("Ferrari", "California"),
    ("Ford", "Excursion"),
    ("BMW", "440i"),
    ("Chevrolet", "HHR"),
    ("INFINITI", "QX56"),
    ("INFINITI", "QX70"),
    ("MAZDA", "MAZDA5"),
    ("Pontiac", "G6"),
    ("Chevrolet", "Cobalt"),
    ("Rivian", "R1T"),
    ("Audi", "S6"),
    ("BMW", "750i"),
    ("BMW", "M240i xDrive"),
    ("BMW", "i8"),
)

MAKE_MODEL_TO_INDEX = {x: i for i, x in enumerate(MAKES_MODELS)}

NUM_MAKE_MODELS = len(MAKE_MODEL_TO_INDEX) + 1
