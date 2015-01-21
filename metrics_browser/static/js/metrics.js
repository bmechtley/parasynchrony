// Model parameter sliders and axis radio buttons.
var params = {
    r: {init: 3.0, min: 0.1, max: 10, symbol: '&lambda;'},
    a: {init: 0.5, min: 0.1, max: 10, symbol: 'a'},
    c: {init: 1.0, min: 0.1, max: 10, symbol: 'c'},
    k: {init: 0.9, min: 0.1, max: 10, symbol: 'k'},
    mh: {init: 0.25, min: 0.1, max: 0.5, symbol: '&mu;<sub>H</sub>'},
    mp: {init: 0.25, min: 0.1, max: 0.5, symbol: '&mu;<sub>P</sub>'},
    Sh: {init: 0.02, min: 0, max: 1.0, symbol: '&Sigma;<sub>H</sub>'},
    Shh: {init: 0.01, min: 0, max: 1.0, symbol: '&Sigma;<sub>HH</sub>'},
    Sp: {init: 0.02, min: 0, max: 1.0, symbol: '&Sigma;<sub>P</sub>'},
    Spp: {init: 0.01, min: 0, max: 1.0, symbol: '&Sigma;<sub>PP</sub>'},
    im: {init: 0, min: -1, max: 1, hidden: true, symbol: 'Im(z)'},
    re: {init: 0, min: -1, max: 1, hidden: true, symbol: 'Re(z)'}
};

var metrics = {
    maxfracsync: {name: 'fraction max synchrony'},
    avgfracsync: {name: 'fraction avg synchrony'},
    xfer: {name: 'transfer function', force_axes: {x: 're', y: 'im'}}
};

var dims = {x: 20, y: 20};

function make_spec(data) {
    console.log(data.data);

    return {
        name: 'plot',
        width: 700,
        height: 700,
        padding: {top: 0, bottom: 0, left: 0, right: 0},
        data: [
            {name: 'tree', format: {type: 'treejson'}, values: data.data},
            {name: 'flattened',  source: 'tree', 'transform': [{type: 'flatten'}]}
        ],
        scales: [
            {
                name: 'spy', type: 'ordinal', range: 'height', zero: true,
                domain: {data: 'tree', field: 'data.var1'}
            }, {
                name: 'spx', type: 'ordinal', range: 'width', zero: true,
                domain: {data: 'tree', field: 'data.var2'}
            }, {
                name: 'fz',
                type: 'linear',
                range: ['#000', '#FFF'],
                domain: {data: 'flattened', field: 'data.value'}
            }
        ],
        marks: [
            {
                type: 'group',
                from: {data: 'tree'},
                properties: {
                    enter: {
                        x: {scale: 'spx', field: 'data.var2'},
                        y: {scale: 'spy', field: 'data.var1'},
                        width: {scale: 'spx', band: true},
                        height: {scale: 'spy', band: true}
                    }
                },
                /*axes: [{type: 'x', scale: 'x'}, {type: 'y', scale: 'y'}],*/
                scales: [
                    {
                        name: 'x', range: 'width', type: 'ordinal', zero: true,
                        domain: {field: 'data.param2'}
                    },
                    {
                        name: 'y', range: 'height', type: 'ordinal', zero: true,
                        domain: {field: 'data.param1'}
                    }
                ],
                marks: [
                    {
                        type: 'rect',
                        properties: {
                            enter: {
                                x: {scale: 'x', field: 'data.param2'},
                                y: {scale: 'y', field: 'data.param1'},
                                width: {scale: 'x', band: true},
                                height: {scale: 'y', band: true}
                            },
                            update: {
                                fill: {scale: 'fz', field: 'data.value'}
                            }
                        }
                    }
                ]
            }
        ]
    };
}

var view;

function selected_metric() {
    return $('select[name=metric] > option:selected:first')
        .attr('value')
}

function selected_axes() {
    return _.object(_.map(['x', 'y'], function(ax) {
        return [
            ax,
            $('input:radio[name=axis_' + ax + ']:checked').attr('value')
        ];
    }));
}

function draw_vis(element, data_uri) {
    $.get(data_uri, function(data) {
        view = vg.parse.spec(make_spec(data), function (chart) {
            chart({el: element}).update();
        });
    });
}

function update_axes() {
    var axes = selected_axes();

    if (axes.hasOwnProperty('force_axes')) {
        $('div[data-prop=value').slider({disabled: false});
    } else {
        $('div[data-prop=value]').each(function () {
            (function (p) {
                p.slider({
                    disabled: !['x', 'y'].every(function (ax) {
                        return !$('input' +
                            '[data-param=' + p.attr('data-param') + ']' +
                            '[data-axis=' + ax + ']'
                        ).prop('checked');
                    })
                });
            })($(this));
        });
    }

    _.each(axes, function(name, ax) {
        console.log(name, ax, params[name]);

        $('input[name=axis_' + ax + '_min]')
            .attr('value', params[name].min);
        $('input[name=axis_' + ax + '_max]')
            .attr('value', params[name].max);
        $('input[name=axis_' + ax + '_n]')
            .attr('value', dims[ax]);
    });
}

$(document).ready(function() {
    $('#params')
        // Sliders and axis radio buttons.
        .append(_.map(params, function(param, name) {
            return $('<div/>', {class: 'paramcontainer'})
                // Label.
                .append($('<label/>', {html: param.symbol}))
                // Slider.
                .append($('<input/>', {
                    type: 'hidden',
                    name: name,
                    value: param.init
                }))
                .append(
                    $('<div/>', {
                        'data-param': name,
                        'data-prop': 'value'
                    }).slider(param).slider({
                        step: (param.max - param.min) / 100.0,
                        value: param.init,
                        change: function(ev) {
                            var param = $(ev.target).attr('data-param');
                            var value = $(ev.target).slider('value');
                            $('input[name=' + param + ']').val(value);

                            draw_vis(
                                '#visualizations',
                                '/pcolor.json?' +
                                    $('#params').serialize()
                            );
                        }
                    }).slider("pips", {
                        step: (param.max - param.min) / 10.0
                    }).slider("float")
                )
                // Axis radio buttons.
                .append(['x', 'y'].map(function (axis) {
                    return $('<input/>', {
                        type: 'radio',
                        'data-param': name,
                        'data-axis': axis,
                        'data-prop': 'axis',
                        name: 'axis_' + axis,
                        value: name
                    }).click(update_axes);
                })).css(
                    'display',
                    param.hasOwnProperty('hidden') ? 'none' : 'block'
                );
        }))
        // Select menu for metric.
        .append($('<label/>', {
            text: 'metric', for: 'metric'
        }))
        .append(
            $('<select/>', {
                name: 'metric',
                'data-prop': 'metric'
            }).append(
                _.map(metrics, function(metric, key) {
                    return $('<option/>', {
                        text: metric.name,
                        value: key
                    })
                })
            ).change(
                function() {
                    var metric = metrics[selected_metric()];
                    var radios = $('input[data-prop=axis]');
                    /* TODO: im/re types are hidden, not radio, so can't be checked */

                    if (metric.hasOwnProperty('force_axes')) {
                        radios.prop('disabled', true);

                        _.each(metric.force_axes, function(name, ax) {
                            $('input[name=axis_' + name +
                                '][value=' + ax + ']'
                            ).prop('checked', true);
                        });
                    } else {
                        radios.prop('disabled', false);
                    }

                    update_axes();
                }
            )
        );
});