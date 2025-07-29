use std::env;
use std::fs;
use serde::{Serialize, Deserialize};
use syn::{File, Item, ItemFn, ItemStruct, ItemEnum, ItemTrait, ItemImpl, ItemMod, ItemUse};
use syn::{Fields, Visibility, Type, FnArg, Pat, ReturnType, TraitItem, ImplItem};

#[derive(Serialize, Deserialize)]
struct Metadata {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<ExtractedData>,
}

#[derive(Serialize, Deserialize)]
struct ExtractedData {
    imports: Vec<ImportInfo>,
    functions: Vec<FunctionInfo>,
    structs: Vec<StructInfo>,
    enums: Vec<EnumInfo>,
    traits: Vec<TraitInfo>,
    impls: Vec<ImplInfo>,
    modules: Vec<ModuleInfo>,
    constants: Vec<ConstantInfo>,
    complexity: usize,
}

#[derive(Serialize, Deserialize)]
struct ImportInfo {
    path: String,
    alias: Option<String>,
    items: Vec<String>,
    line: usize,
}

#[derive(Serialize, Deserialize)]
struct FunctionInfo {
    name: String,
    is_pub: bool,
    is_async: bool,
    is_unsafe: bool,
    is_const: bool,
    params: Vec<ParamInfo>,
    return_type: Option<String>,
    generics: Vec<String>,
    line: usize,
}

#[derive(Serialize, Deserialize)]
struct ParamInfo {
    name: String,
    param_type: String,
    is_mut: bool,
}

#[derive(Serialize, Deserialize)]
struct StructInfo {
    name: String,
    is_pub: bool,
    fields: Vec<FieldInfo>,
    generics: Vec<String>,
    derives: Vec<String>,
    line: usize,
}

#[derive(Serialize, Deserialize)]
struct FieldInfo {
    name: Option<String>,
    field_type: String,
    is_pub: bool,
}

#[derive(Serialize, Deserialize)]
struct EnumInfo {
    name: String,
    is_pub: bool,
    variants: Vec<VariantInfo>,
    generics: Vec<String>,
    derives: Vec<String>,
    line: usize,
}

#[derive(Serialize, Deserialize)]
struct VariantInfo {
    name: String,
    fields: Vec<FieldInfo>,
}

#[derive(Serialize, Deserialize)]
struct TraitInfo {
    name: String,
    is_pub: bool,
    methods: Vec<MethodInfo>,
    generics: Vec<String>,
    line: usize,
}

#[derive(Serialize, Deserialize)]
struct MethodInfo {
    name: String,
    params: Vec<ParamInfo>,
    return_type: Option<String>,
    is_async: bool,
    is_unsafe: bool,
}

#[derive(Serialize, Deserialize)]
struct ImplInfo {
    trait_name: Option<String>,
    for_type: String,
    methods: Vec<FunctionInfo>,
    line: usize,
}

#[derive(Serialize, Deserialize)]
struct ModuleInfo {
    name: String,
    is_pub: bool,
    line: usize,
}

#[derive(Serialize, Deserialize)]
struct ConstantInfo {
    name: String,
    is_pub: bool,
    const_type: Option<String>,
    line: usize,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        let result = Metadata {
            success: false,
            error: Some("Usage: rust_ast_parser <file_path>".to_string()),
            data: None,
        };
        println!("{}", serde_json::to_string(&result).unwrap());
        std::process::exit(1);
    }

    let filename = &args[1];
    
    match fs::read_to_string(filename) {
        Ok(content) => {
            match syn::parse_file(&content) {
                Ok(file) => {
                    let data = extract_metadata(&file);
                    let result = Metadata {
                        success: true,
                        error: None,
                        data: Some(data),
                    };
                    println!("{}", serde_json::to_string(&result).unwrap());
                }
                Err(e) => {
                    let result = Metadata {
                        success: false,
                        error: Some(format!("Failed to parse file: {}", e)),
                        data: None,
                    };
                    println!("{}", serde_json::to_string(&result).unwrap());
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            let result = Metadata {
                success: false,
                error: Some(format!("Failed to read file: {}", e)),
                data: None,
            };
            println!("{}", serde_json::to_string(&result).unwrap());
            std::process::exit(1);
        }
    }
}

fn extract_metadata(file: &File) -> ExtractedData {
    let mut data = ExtractedData {
        imports: Vec::new(),
        functions: Vec::new(),
        structs: Vec::new(),
        enums: Vec::new(),
        traits: Vec::new(),
        impls: Vec::new(),
        modules: Vec::new(),
        constants: Vec::new(),
        complexity: 1,
    };

    for (idx, item) in file.items.iter().enumerate() {
        let line = idx + 1; // Approximate line number
        
        match item {
            Item::Use(item_use) => {
                data.imports.push(extract_import(item_use, line));
            }
            Item::Fn(item_fn) => {
                let func_info = extract_function(item_fn, line);
                data.complexity += calculate_function_complexity(item_fn);
                data.functions.push(func_info);
            }
            Item::Struct(item_struct) => {
                data.structs.push(extract_struct(item_struct, line));
            }
            Item::Enum(item_enum) => {
                data.enums.push(extract_enum(item_enum, line));
            }
            Item::Trait(item_trait) => {
                data.traits.push(extract_trait(item_trait, line));
            }
            Item::Impl(item_impl) => {
                data.impls.push(extract_impl(item_impl, line));
            }
            Item::Mod(item_mod) => {
                data.modules.push(extract_module(item_mod, line));
            }
            Item::Const(item_const) => {
                data.constants.push(ConstantInfo {
                    name: item_const.ident.to_string(),
                    is_pub: matches!(item_const.vis, Visibility::Public(_)),
                    const_type: Some(type_to_string(&item_const.ty)),
                    line,
                });
            }
            _ => {}
        }
    }

    data
}

fn extract_import(item_use: &ItemUse, line: usize) -> ImportInfo {
    let path = quote::quote!(#item_use).to_string();
    ImportInfo {
        path: path.clone(),
        alias: None,
        items: Vec::new(),
        line,
    }
}

fn extract_function(item_fn: &ItemFn, line: usize) -> FunctionInfo {
    let mut params = Vec::new();
    
    for input in &item_fn.sig.inputs {
        if let FnArg::Typed(pat_type) = input {
            let name = pat_to_string(&pat_type.pat);
            let param_type = type_to_string(&pat_type.ty);
            params.push(ParamInfo {
                name,
                param_type,
                is_mut: false, // Simplified
            });
        }
    }

    let return_type = match &item_fn.sig.output {
        ReturnType::Default => None,
        ReturnType::Type(_, ty) => Some(type_to_string(ty)),
    };

    let generics: Vec<String> = item_fn.sig.generics.params.iter()
        .map(|p| quote::quote!(#p).to_string())
        .collect();

    FunctionInfo {
        name: item_fn.sig.ident.to_string(),
        is_pub: matches!(item_fn.vis, Visibility::Public(_)),
        is_async: item_fn.sig.asyncness.is_some(),
        is_unsafe: item_fn.sig.unsafety.is_some(),
        is_const: item_fn.sig.constness.is_some(),
        params,
        return_type,
        generics,
        line,
    }
}

fn extract_struct(item_struct: &ItemStruct, line: usize) -> StructInfo {
    let mut fields = Vec::new();
    
    match &item_struct.fields {
        Fields::Named(fields_named) => {
            for field in &fields_named.named {
                fields.push(FieldInfo {
                    name: field.ident.as_ref().map(|i| i.to_string()),
                    field_type: type_to_string(&field.ty),
                    is_pub: matches!(field.vis, Visibility::Public(_)),
                });
            }
        }
        Fields::Unnamed(fields_unnamed) => {
            for field in &fields_unnamed.unnamed {
                fields.push(FieldInfo {
                    name: None,
                    field_type: type_to_string(&field.ty),
                    is_pub: matches!(field.vis, Visibility::Public(_)),
                });
            }
        }
        Fields::Unit => {}
    }

    let generics: Vec<String> = item_struct.generics.params.iter()
        .map(|p| quote::quote!(#p).to_string())
        .collect();

    let derives = extract_derives(&item_struct.attrs);

    StructInfo {
        name: item_struct.ident.to_string(),
        is_pub: matches!(item_struct.vis, Visibility::Public(_)),
        fields,
        generics,
        derives,
        line,
    }
}

fn extract_enum(item_enum: &ItemEnum, line: usize) -> EnumInfo {
    let mut variants = Vec::new();
    
    for variant in &item_enum.variants {
        let mut fields = Vec::new();
        
        match &variant.fields {
            Fields::Named(fields_named) => {
                for field in &fields_named.named {
                    fields.push(FieldInfo {
                        name: field.ident.as_ref().map(|i| i.to_string()),
                        field_type: type_to_string(&field.ty),
                        is_pub: matches!(field.vis, Visibility::Public(_)),
                    });
                }
            }
            Fields::Unnamed(fields_unnamed) => {
                for field in &fields_unnamed.unnamed {
                    fields.push(FieldInfo {
                        name: None,
                        field_type: type_to_string(&field.ty),
                        is_pub: matches!(field.vis, Visibility::Public(_)),
                    });
                }
            }
            Fields::Unit => {}
        }
        
        variants.push(VariantInfo {
            name: variant.ident.to_string(),
            fields,
        });
    }

    let generics: Vec<String> = item_enum.generics.params.iter()
        .map(|p| quote::quote!(#p).to_string())
        .collect();

    let derives = extract_derives(&item_enum.attrs);

    EnumInfo {
        name: item_enum.ident.to_string(),
        is_pub: matches!(item_enum.vis, Visibility::Public(_)),
        variants,
        generics,
        derives,
        line,
    }
}

fn extract_trait(item_trait: &ItemTrait, line: usize) -> TraitInfo {
    let mut methods = Vec::new();
    
    for item in &item_trait.items {
        if let TraitItem::Method(method) = item {
            let mut params = Vec::new();
            
            for input in &method.sig.inputs {
                if let FnArg::Typed(pat_type) = input {
                    let name = pat_to_string(&pat_type.pat);
                    let param_type = type_to_string(&pat_type.ty);
                    params.push(ParamInfo {
                        name,
                        param_type,
                        is_mut: false,
                    });
                }
            }

            let return_type = match &method.sig.output {
                ReturnType::Default => None,
                ReturnType::Type(_, ty) => Some(type_to_string(ty)),
            };

            methods.push(MethodInfo {
                name: method.sig.ident.to_string(),
                params,
                return_type,
                is_async: method.sig.asyncness.is_some(),
                is_unsafe: method.sig.unsafety.is_some(),
            });
        }
    }

    let generics: Vec<String> = item_trait.generics.params.iter()
        .map(|p| quote::quote!(#p).to_string())
        .collect();

    TraitInfo {
        name: item_trait.ident.to_string(),
        is_pub: matches!(item_trait.vis, Visibility::Public(_)),
        methods,
        generics,
        line,
    }
}

fn extract_impl(item_impl: &ItemImpl, line: usize) -> ImplInfo {
    let mut methods = Vec::new();
    
    for item in &item_impl.items {
        if let ImplItem::Method(method) = item {
            methods.push(extract_function_from_method(method, line));
        }
    }

    let trait_name = item_impl.trait_.as_ref().map(|(_, path, _)| {
        quote::quote!(#path).to_string()
    });

    ImplInfo {
        trait_name,
        for_type: type_to_string(&item_impl.self_ty),
        methods,
        line,
    }
}

fn extract_function_from_method(method: &syn::ImplItemMethod, line: usize) -> FunctionInfo {
    let mut params = Vec::new();
    
    for input in &method.sig.inputs {
        if let FnArg::Typed(pat_type) = input {
            let name = pat_to_string(&pat_type.pat);
            let param_type = type_to_string(&pat_type.ty);
            params.push(ParamInfo {
                name,
                param_type,
                is_mut: false,
            });
        }
    }

    let return_type = match &method.sig.output {
        ReturnType::Default => None,
        ReturnType::Type(_, ty) => Some(type_to_string(ty)),
    };

    let generics: Vec<String> = method.sig.generics.params.iter()
        .map(|p| quote::quote!(#p).to_string())
        .collect();

    FunctionInfo {
        name: method.sig.ident.to_string(),
        is_pub: matches!(method.vis, Visibility::Public(_)),
        is_async: method.sig.asyncness.is_some(),
        is_unsafe: method.sig.unsafety.is_some(),
        is_const: method.sig.constness.is_some(),
        params,
        return_type,
        generics,
        line,
    }
}

fn extract_module(item_mod: &ItemMod, line: usize) -> ModuleInfo {
    ModuleInfo {
        name: item_mod.ident.to_string(),
        is_pub: matches!(item_mod.vis, Visibility::Public(_)),
        line,
    }
}

fn extract_derives(attrs: &[syn::Attribute]) -> Vec<String> {
    let mut derives = Vec::new();
    
    for attr in attrs {
        if attr.path.is_ident("derive") {
            if let Ok(syn::Meta::List(list)) = attr.parse_meta() {
                for nested in &list.nested {
                    if let syn::NestedMeta::Meta(syn::Meta::Path(path)) = nested {
                        if let Some(ident) = path.get_ident() {
                            derives.push(ident.to_string());
                        }
                    }
                }
            }
        }
    }
    
    derives
}

fn calculate_function_complexity(item_fn: &ItemFn) -> usize {
    // Simplified complexity calculation
    let block = &item_fn.block;
    let code = quote::quote!(#block).to_string();
    
    let mut complexity = 1;
    
    // Count control flow keywords
    for keyword in &["if", "match", "for", "while", "loop"] {
        complexity += code.matches(keyword).count();
    }
    
    complexity
}

fn type_to_string(ty: &Type) -> String {
    quote::quote!(#ty).to_string()
}

fn pat_to_string(pat: &Pat) -> String {
    match pat {
        Pat::Ident(pat_ident) => pat_ident.ident.to_string(),
        _ => quote::quote!(#pat).to_string(),
    }
}